import argparse
import pymimir as mm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import Dict, List, Tuple
from utils import create_device, get_atom_name, get_atoms, get_goal, get_predicate_name, relations_to_tensors, save_checkpoint, load_checkpoint


class StateSampler:
    def __init__(self, state_spaces: List[mm.StateSpace]) -> None:
        self._state_spaces = state_spaces
        self._max_distances = []
        self._has_deadends = []
        self._deadend_distance = float('inf')
        for state_space in state_spaces:
            max_goal_distance = 0
            has_deadend = False
            for goal_distance in state_space.get_goal_distances():
                if goal_distance != self._deadend_distance:
                    max_goal_distance = max(max_goal_distance, int(goal_distance))
                else:
                    has_deadend = True
            self._max_distances.append(max_goal_distance)
            self._has_deadends.append(has_deadend)

    def sample(self) -> Tuple[mm.State, mm.StateSpace, int]:
        # To achieve an even distribution, we uniformly sample a state space and select a valid goal-distance within that space.
        # Finally, we randomly sample a state from the selected state space and with the goal-distance.
        state_space_index = random.randint(0, len(self._state_spaces) - 1)
        sampled_state_space = self._state_spaces[state_space_index]
        max_goal_distance = self._max_distances[state_space_index]
        has_deadends = self._has_deadends[state_space_index]
        goal_distance = random.randint(-1 if has_deadends else 0, max_goal_distance)
        if goal_distance < 0:
            sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(self._deadend_distance)
        else:
            sampled_state_index = sampled_state_space.sample_vertex_index_with_goal_distance(goal_distance)
        sampled_state = sampled_state_space.get_vertex(sampled_state_index)
        return (sampled_state.get_state(), sampled_state_space, goal_distance)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument('--input', required=True, type=Path, help='Path to the training dataset')
    parser.add_argument('--model', default=None, type=Path, help='Path to a pre-trained model to continue training from')
    parser.add_argument('--embedding_size', default=32, type=int, help='Dimension of the embedding vector for each object')
    parser.add_argument('--layers', default=30, type=int, help='Number of layers in the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per batch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate for the training process')
    parser.add_argument('--num_epochs', default=1_000, type=int, help='Number of epochs for the training process')
    args = parser.parse_args()
    return args


def _parse_instances(input: Path) -> Tuple[str, List[str]]:
    print('Parsing files...')
    if input.is_file():
        domain_file = str(input.parent / 'domain.pddl')
        problem_files = [str(input)]
    else:
        domain_file = str(input / 'domain.pddl')
        problem_files = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_files.sort()
    return domain_file, problem_files


def _generate_state_spaces(domain_path: str, problem_paths: List[str]) -> List[mm.StateSpace]:
    print('Generating state spaces...')
    state_spaces: List[mm.StateSpace] = []
    for problem_path in problem_paths:
        print(f'> Expanding: {problem_path}')
        state_space = mm.StateSpace.create(domain_path, problem_path, mm.StateSpaceOptions(max_num_states=1_000_000, timeout_ms=60_000))
        if state_space is not None:
            state_spaces.append(state_space)
            print(f'- # States: {state_space.get_num_vertices()}')
        else:
            print('- Skipped')
    state_spaces.sort(key=lambda state_space: state_space.get_num_vertices())
    return state_spaces


def _create_state_samplers(state_spaces: List[mm.StateSpace]) -> Tuple[StateSampler, StateSampler]:
    print('Creating state samplers...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSampler(train_state_spaces)
    validation_dataset = StateSampler(validation_state_spaces)
    return train_dataset, validation_dataset


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int, device: torch.device) -> nn.Module:
    predicates = []
    predicates.extend(domain.get_static_predicates())
    predicates.extend(domain.get_fluent_predicates())
    predicates.extend(domain.get_derived_predicates())
    relation_name_arities = [(get_predicate_name(predicate, False, True), len(predicate.get_parameters())) for predicate in predicates]
    relation_name_arities.extend([(get_predicate_name(predicate, True, True), len(predicate.get_parameters())) for predicate in predicates])
    relation_name_arities.extend([(get_predicate_name(predicate, True, False), len(predicate.get_parameters())) for predicate in predicates])
    model = SmoothmaxRelationalNeuralNetwork(relation_name_arities, embedding_size, num_layers).to(device)
    return model


def _sample_state_to_batch(relations: Dict[str, List[int]], sizes: List[int], targets: List[int], states: StateSampler):
    state, state_space, target = states.sample()
    offset = sum(sizes)
    # Helper function for populating relations and sizes.
    def add_relations(atom, is_goal_atom):
        predicate_name = get_atom_name(atom, state, is_goal_atom)
        term_ids = [term.get_index() + offset for term in atom.get_objects()]
        if predicate_name not in relations: relations[predicate_name] = term_ids
        else: relations[predicate_name].extend(term_ids)
    # Add state to relations and sizes, together with the goal.
    for atom in get_atoms(state, state_space.get_problem(), state_space.get_pddl_repositories()): add_relations(atom, False)
    for atom in get_goal(state_space.get_problem()): add_relations(atom, True)
    sizes.append(len(state_space.get_problem().get_objects()))
    targets.append(target)


def _sample_batch(states: StateSampler, batch_size: int, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    relations = {}
    sizes = []
    targets = []
    for _ in range(batch_size):
        _sample_state_to_batch(relations, sizes, targets, states)
    relation_tensors = relations_to_tensors(relations, device)
    size_tensor = torch.tensor(sizes, dtype=torch.int, device=device, requires_grad=False)
    target_tensor = torch.tensor(targets, dtype=torch.float, device=device, requires_grad=False)
    return relation_tensors, size_tensor, target_tensor


def _train(model: SmoothmaxRelationalNeuralNetwork,
           optimizer: optim.Adam,
           train_states: StateSampler,
           validation_states: StateSampler,
           num_epochs: int,
           batch_size: int,
           device: torch.device) -> None:
    # While we can sample states on the fly from the state spaces, this creates
    # a significant overhead because the states need to be translated to the
    # correct format and transferred to the GPU. Instead, we sample a fixed
    # number of states and move them to the GPU before training. This approach
    # increases GPU utilization.
    print('Creating datasets...')
    train_dataset = [_sample_batch(train_states, batch_size, device) for _ in range(10_000)]
    validation_dataset = [_sample_batch(validation_states, batch_size, device) for _ in range(1_000)]
    # Training loop
    best_absolute_error = None  # Track the best validation loss to detect overfitting.
    print('Training model...')
    for epoch in range(0, num_epochs):
        # Train step
        for index, (relations, sizes, targets) in enumerate(train_dataset):
            # Forward pass
            value_predictions, deadend_predictions = model.forward(relations, sizes)
            # The value loss has two parts: a standard absolute error loss
            # and a distance loss. The distance loss compares the predicted
            # values. Specifically, for two states, s and s', the loss states
            # that |V(s) - V(s')| should equal |V*(s) - V*(s')|. This is done
            # for all possible pairs of s and s' in the batch.
            value_mask = targets.ge(0)
            value_loss = ((value_predictions - targets).abs() * value_mask).mean()
            prediction_pairs = torch.cartesian_prod(value_predictions, value_predictions)
            target_pairs = torch.cartesian_prod(targets, targets)
            prediction_distances = (prediction_pairs[:, 0] - prediction_pairs[:, 1]).abs()
            target_distances = (target_pairs[:, 0] - target_pairs[:, 1]).abs()
            distance_loss = (prediction_distances - target_distances).abs().mean()
            value_loss = value_loss + distance_loss
            # The deadend loss is simply binary cross entropy with logits.
            deadend_targets = 1.0 * targets.less(0)
            deadend_loss = torch.nn.functional.binary_cross_entropy_with_logits(deadend_predictions, deadend_targets)
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss = value_loss + deadend_loss
            total_loss.backward()
            optimizer.step()
            # Print loss every 100 steps (printing every step forces synchronization with CPU)
            if (index + 1) % 100 == 0:
                print(f'[{epoch + 1}/{num_epochs}; {index + 1}/{len(train_dataset)}] Loss: {total_loss.item():.4f}')
        # Validation step
        with torch.no_grad():
            absolute_error = torch.zeros([1], dtype=torch.float, device=device)
            deadend_error = torch.zeros([1], dtype=torch.float, device=device)
            for relations, sizes, targets in validation_dataset:
                value_predictions, deadend_predictions = model.forward(relations, sizes)
                value_mask = targets.ge(0)
                absolute_error += ((value_predictions - targets).abs() * value_mask).sum()
                deadend_targets = 1.0 * targets.less(0)
                deadend_error += torch.nn.functional.binary_cross_entropy_with_logits(deadend_predictions, deadend_targets, reduction='sum')
            total_samples = len(validation_dataset) * batch_size
            absolute_error = absolute_error / total_samples
            deadend_error = deadend_error / total_samples
            print(f'[{epoch + 1}/{num_epochs}] Absolute error: {absolute_error.item():.4f}; Deadend error: {deadend_error.item():.4f}')
            save_checkpoint(model, optimizer, 'latest.pth')
            if (best_absolute_error is None) or (absolute_error < best_absolute_error):
                best_absolute_error = absolute_error
                save_checkpoint(model, optimizer, 'best.pth')
                print(f'[{epoch + 1}/{num_epochs}] Saved new best model')


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    device = create_device()
    domain_path, problem_paths = _parse_instances(args.input)
    state_spaces = _generate_state_spaces(domain_path, problem_paths)
    train_dataset, validation_dataset = _create_state_samplers(state_spaces)
    domain = state_spaces[0].get_problem().get_domain()
    if args.model is None:
        print('Creating a new model and optimizer...')
        model = _create_model(domain, args.embedding_size, args.layers, device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        print(f'Loading an existing model and optimizer... ({args.model})')
        model, optimizer = load_checkpoint(args.model, device)
    _train(model, optimizer, train_dataset, validation_dataset, args.num_epochs, args.batch_size, device)


if __name__ == "__main__":
    _main(_parse_arguments())
