import argparse
import pymimir as mm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from rgnn import RelationalGraphNeuralNetwork, load_checkpoint, save_checkpoint
from utils import StateWrapper, create_device


class StateSampler:
    def __init__(self, state_spaces: list[mm.StateSpace]) -> None:
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

    def sample(self) -> tuple[mm.State, mm.StateSpace, int]:
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
    parser.add_argument('--aggregation', default='smax', type=str, help='Aggregation function ("smax", "max", "sum", or "mean")')
    parser.add_argument('--layers', default=30, type=int, help='Number of layers in the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per batch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate for the training process')
    parser.add_argument('--num_epochs', default=1_000, type=int, help='Number of epochs for the training process')
    args = parser.parse_args()
    return args


def _parse_instances(input: Path) -> tuple[str, list[str]]:
    print('Parsing files...')
    if input.is_file():
        domain_file = str(input.parent / 'domain.pddl')
        problem_files = [str(input)]
    else:
        domain_file = str(input / 'domain.pddl')
        problem_files = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_files.sort()
    return domain_file, problem_files


def _generate_state_spaces(domain_path: str, problem_paths: list[str]) -> list[mm.StateSpace]:
    print('Generating state spaces...')
    state_spaces: list[mm.StateSpace] = []
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


def _create_state_samplers(state_spaces: list[mm.StateSpace]) -> tuple[StateSampler, StateSampler]:
    print('Creating state samplers...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSampler(train_state_spaces)
    validation_dataset = StateSampler(validation_state_spaces)
    return train_dataset, validation_dataset


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int, aggregation: str, device: torch.device) -> nn.Module:
    return RelationalGraphNeuralNetwork(domain, embedding_size, num_layers, aggregation).to(device)


def _sample_batch(state_sampler: StateSampler, batch_size: int, device: torch.device) -> tuple[list[StateWrapper], torch.Tensor]:
    states: list[StateWrapper] = []
    targets: list[float]  = []
    for _ in range(batch_size):
        state, state_space, target = state_sampler.sample()
        states.append(StateWrapper(state, state_space.get_problem(), state_space.get_pddl_repositories()))
        targets.append(target)
    return states, torch.tensor(targets,  dtype=torch.float, requires_grad=False, device=device)


def _train(model: RelationalGraphNeuralNetwork,
           optimizer: optim.Adam,
           train_states: StateSampler,
           validation_states: StateSampler,
           num_epochs: int,
           batch_size: int,
           device: torch.device) -> None:
    TRAIN_SIZE = 1000
    VALIDATION_SIZE = 100
    # Training loop
    best_absolute_error = None  # Track the best validation loss to detect overfitting.
    print('Training model...')
    for epoch in range(0, num_epochs):
        # Train step
        for index in range(TRAIN_SIZE):
            # Forward pass
            states, value_targets = _sample_batch(train_states, batch_size, device)
            value_predictions, deadend_predictions = model.forward(states)
            # Value loss is absolute mean error.
            value_mask = value_targets.ge(0)
            value_loss = ((value_predictions - value_targets).abs() * value_mask).mean()
            # The deadend loss is simply binary cross entropy with logits.
            deadend_targets = 1.0 * value_targets.less(0)
            deadend_loss = torch.nn.functional.binary_cross_entropy_with_logits(deadend_predictions, deadend_targets)
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss = value_loss + deadend_loss
            total_loss.backward()
            optimizer.step()
            # Print loss every 100 steps (printing every step forces synchronization with CPU)
            if (index + 1) % 100 == 0:
                print(f'[{epoch + 1}/{num_epochs}; {index + 1}/{TRAIN_SIZE}] Loss: {total_loss.item():.4f}')
        # Validation step
        with torch.no_grad():
            absolute_error = torch.zeros([1], dtype=torch.float, device=device)
            deadend_error = torch.zeros([1], dtype=torch.float, device=device)
            for index in range(VALIDATION_SIZE):
                states, value_targets = _sample_batch(validation_states, batch_size, device)
                value_predictions, deadend_predictions = model.forward(states)
                value_mask = value_targets.ge(0)
                absolute_error += ((value_predictions - value_targets).abs() * value_mask).sum()
                deadend_targets = 1.0 * value_targets.less(0)
                deadend_error += torch.nn.functional.binary_cross_entropy_with_logits(deadend_predictions, deadend_targets, reduction='sum')
            total_samples = VALIDATION_SIZE * batch_size
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
        model = _create_model(domain, args.embedding_size, args.layers, args.aggregation, device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        print(f'Loading an existing model and optimizer... ({args.model})')
        model, optimizer = load_checkpoint(domain, args.model, device)
    _train(model, optimizer, train_dataset, validation_dataset, args.num_epochs, args.batch_size, device)


if __name__ == "__main__":
    _main(_parse_arguments())
