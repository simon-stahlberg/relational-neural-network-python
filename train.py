import argparse
import pymimir as mm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import Dict, List, Tuple, Union
from utils import create_device, save_checkpoint, load_checkpoint


class StateSampler:
    def __init__(self, state_spaces: List[mm.StateSpace]) -> None:
        self._state_spaces = state_spaces

    def sample(self) -> Tuple[mm.State, mm.StateSpace, int]:
        # To achieve an even distribution, we uniformly sample a state space and select a valid goal-distance within that space.
        # Finally, we randomly sample a state from the selected state space and with the goal-distance.
        state_space_index = random.randint(0, len(self._state_spaces) - 1)
        sampled_state_space = self._state_spaces[state_space_index]
        longest_distance = sampled_state_space.get_max_goal_distance()
        goal_distance = random.randint(0, longest_distance)
        sampled_state = sampled_state_space.sample_state_with_goal_distance(goal_distance)
        return (sampled_state, sampled_state_space, goal_distance)



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
        state_space = mm.StateSpace.create(domain_path, problem_path, 1_000_000, 60_000)
        if state_space is not None:
            state_spaces.append(state_space)
            print(f'- # States: {state_space.get_num_states()}')
        else:
            print('- Skipped')
    state_spaces.sort(key=lambda state_space: state_space.get_num_states())
    return state_spaces


def _create_state_samplers(state_spaces: List[mm.StateSpace]) -> Tuple[StateSampler, StateSampler]:
    print('Creating state samplers...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSampler(train_state_spaces)
    validation_dataset = StateSampler(validation_state_spaces)
    return train_dataset, validation_dataset


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int) -> nn.Module:
    predicates = domain.get_static_predicates() + domain.get_fluent_predicates() + domain.get_derived_predicates()
    relation_name_arities = [('relation_' + predicate.get_name(), len(predicate.get_parameters())) for predicate in predicates]
    relation_name_arities.extend([('relation_' + predicate.get_name() + '_goal_true', len(predicate.get_parameters())) for predicate in predicates])
    relation_name_arities.extend([('relation_' + predicate.get_name() + '_goal_false', len(predicate.get_parameters())) for predicate in predicates])
    model = SmoothmaxRelationalNeuralNetwork(relation_name_arities, embedding_size, num_layers)
    return model


def _get_atoms(state: mm.State, state_space: mm.StateSpace) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    problem = state_space.get_problem()
    factories = state_space.get_factories()
    static_atoms = [literal.get_atom() for literal in problem.get_static_initial_literals()]
    fluent_atoms = factories.get_fluent_ground_atoms_from_ids(state.get_fluent_atoms())
    derived_atoms = factories.get_derived_ground_atoms_from_ids(state.get_derived_atoms())
    all_atoms = static_atoms + fluent_atoms + derived_atoms
    return all_atoms


def _get_goal(state_space: mm.StateSpace) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    problem = state_space.get_problem()
    static_goal = [literal.get_atom() for literal in problem.get_static_goal_condition()]
    fluent_goal = [literal.get_atom() for literal in problem.get_fluent_goal_condition()]
    derived_goal = [literal.get_atom() for literal in problem.get_derived_goal_condition()]
    full_goal = static_goal + fluent_goal + derived_goal
    return full_goal


def _relations_to_tensors(term_id_groups: Dict[str, List[int]], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


def _sample_state_to_batch(batch_relations: Dict[str, List[int]], batch_sizes: List[int], batch_targets: List[int], states: StateSampler):
    state, state_space, target_value = states.sample()
    id_offset = sum(batch_sizes)
    state_atoms = _get_atoms(state, state_space)
    goal_atoms = _get_goal(state_space)
    for atom in state_atoms:
        predicate_name = 'relation_' + atom.get_predicate().get_name()
        term_ids = [term.get_identifier() + id_offset for term in atom.get_objects()]
        if predicate_name not in batch_relations: batch_relations[predicate_name] = term_ids
        else: batch_relations[predicate_name].extend(term_ids)
    for atom in goal_atoms:
        predicate_name = ('relation_' + atom.get_predicate().get_name() + '_goal') + ('_true' if state.contains(atom) else '_false')
        term_ids = [term.get_identifier() + id_offset for term in atom.get_objects()]
        if predicate_name not in batch_relations: batch_relations[predicate_name] = term_ids
        else: batch_relations[predicate_name].extend(term_ids)
    batch_sizes.append(len(state_space.get_problem().get_objects()))
    batch_targets.append(target_value)


def _sample_batch(states: StateSampler, batch_size: int, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch_relations = {}
    batch_sizes = []
    batch_targets = []
    for _ in range(batch_size):
        _sample_state_to_batch(batch_relations, batch_sizes, batch_targets, states)
    batch_relation_tensors = _relations_to_tensors(batch_relations, device)
    batch_size_tensor = torch.tensor(batch_sizes, dtype=torch.int, device=device, requires_grad=False)
    batch_target_tensor = torch.tensor(batch_targets, dtype=torch.float, device=device, requires_grad=False)
    return batch_relation_tensors, batch_size_tensor, batch_target_tensor


def _train(model: SmoothmaxRelationalNeuralNetwork, train_states: StateSampler, validation_states: StateSampler, num_epochs: int, batch_size: int) -> None:
    device = create_device()
    model = model.to(device)
    # While we can sample states on the fly from the state spaces, this creates
    # a significant overhead because the states need to be translated to the
    # correct format and transferred to the GPU. Instead, we sample a fixed
    # number of states and move them to the GPU before training. This approach
    # increases GPU utilization.
    print('Creating datasets...')
    train_dataset = [_sample_batch(train_states, batch_size, device) for _ in range(10_000)]
    validation_dataset = [_sample_batch(validation_states, batch_size, device) for _ in range(1_000)]
    # Training loop
    best_validation_loss = None  # Track the best validation loss to detect overfitting.
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    print('Training model...')
    for epoch in range(0, num_epochs):
        # Train step
        for index, (relations, sizes, targets) in enumerate(train_dataset):
            # Forward pass
            predictions = model.forward(relations, sizes).view(-1)
            loss = (predictions - targets).abs().mean()  # MAE
            # loss = (predictions - targets).square().mean()  # MSE
            # loss = (predictions - targets).square().mean().sqrt()  # RMSE
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print loss every 100 steps (printing every step forces synchronization with CPU)
            if (index + 1) % 100 == 0:
                print(f'[{epoch + 1}/{num_epochs}; {index + 1}/{len(train_dataset)}] Loss: {loss.item():.4f}')
        # Validation step
        with torch.no_grad():
            total_square_error = torch.zeros([1], dtype=torch.float, device=device)
            total_absolute_error = torch.zeros([1], dtype=torch.float, device=device)
            for index, (relations, sizes, targets) in enumerate(validation_dataset):
                predictions = model.forward(relations, sizes).view(-1)
                total_square_error += (predictions - targets).square().sum()
                total_absolute_error += (predictions - targets).abs().sum()
            total_samples = len(validation_dataset) * batch_size
            validation_loss = total_absolute_error / total_samples
            print(f'[{epoch + 1}/{num_epochs}] Validation loss: {validation_loss.item():.4f}')
            if (best_validation_loss is None) or (validation_loss < best_validation_loss):
                best_validation_loss = validation_loss
                save_checkpoint(model, optimizer, "best.pth")
                print(f'[{epoch + 1}/{num_epochs}] Saved new best model')


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    domain_path, problem_paths = _parse_instances(args.input)
    state_spaces = _generate_state_spaces(domain_path, problem_paths)
    train_dataset, validation_dataset = _create_state_samplers(state_spaces)
    domain = state_spaces[0].get_problem().get_domain()
    model = _create_model(domain, args.embedding_size, args.layers)
    _train(model, train_dataset, validation_dataset, args.num_epochs, args.batch_size)


if __name__ == "__main__":
    _main(_parse_arguments())
