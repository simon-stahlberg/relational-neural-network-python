import argparse
import pymimir as mm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from rgnn import RelationalGraphNeuralNetwork
from utils import create_device


class StateDataset:
    def __init__(self, state_space_samplers: list[mm.StateSpaceSampler]) -> None:
        self._state_spaces = state_space_samplers

    def sample(self) -> tuple[mm.State, mm.StateLabel]:
        # To achieve a good distribution, we uniformly sample a state space and select a valid goal-distance within that space.
        # Finally, we randomly sample a state from the selected state space and with the goal-distance.
        state_space_index = random.randint(0, len(self._state_spaces) - 1)
        state_space = self._state_spaces[state_space_index]
        lower_bound = -1 if (state_space.num_dead_end_states() > 0) else 0
        upper_bound = state_space.max_steps_to_goal()
        goal_distance = random.randint(lower_bound, upper_bound)  # -1 means dead-end state, 0 means goal state, and positive values mean distance to goal.
        sampled_state = state_space.sample_dead_end_state() if goal_distance < 0 else state_space.sample_state_n_steps_from_goal(goal_distance)
        sampled_label = state_space.get_state_label(sampled_state)
        return (sampled_state, sampled_label)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument('--input', required=True, type=Path, help='Path to the training dataset')
    parser.add_argument('--model', default=None, type=Path, help='Path to a pre-trained model to continue training from')
    parser.add_argument('--embedding_size', default=32, type=int, help='Dimension of the embedding vector for each object')
    parser.add_argument('--aggregation', default='hmax', type=str, help='Aggregation function ("smax", "hmax", "sum", or "mean")')
    parser.add_argument('--layers', default=30, type=int, help='Number of layers in the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per batch')
    parser.add_argument('--learning_rate', default=0.00002, type=float, help='Learning rate for the training process')
    parser.add_argument('--num_epochs', default=1_000, type=int, help='Number of epochs for the training process')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args


def _get_instance_paths(input: Path) -> tuple[str, list[str]]:
    print('Finding paths...')
    if input.is_file():
        domain_file = str(input.parent / 'domain.pddl')
        problem_files = [str(input)]
    else:
        domain_file = str(input / 'domain.pddl')
        problem_files = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_files.sort()
    return domain_file, problem_files


def _parse_instances(domain_path: str, problem_paths: list[str]) -> tuple[mm.Domain, list[mm.Problem]]:
    print('Parsing instances...')
    domain = mm.Domain(domain_path)
    problems = [mm.Problem(domain, problem_path) for problem_path in problem_paths]
    print(f'- Domain: {domain.get_name()}')
    print(f'- # Problems: {len(problems)}')
    return domain, problems


def _generate_state_spaces(problems: list[mm.Problem], seed: int) -> list[mm.StateSpaceSampler]:
    print('Generating state spaces...')
    state_space_samplers: list[mm.StateSpaceSampler] = []
    for problem in problems:
        print(f'> Expanding: {problem.get_name()}')
        state_space_sampler = mm.StateSpaceSampler.new(problem, 10_000_000)
        if state_space_sampler is not None:
            state_space_sampler.set_seed(seed)
            state_space_samplers.append(state_space_sampler)
            print(f'- Added with {state_space_sampler.num_states()} states')
        else:
            print('- Skipped')
    return state_space_samplers


def _create_datasets(state_space_samplers: list[mm.StateSpaceSampler]) -> tuple[StateDataset, StateDataset]:
    print('Creating state samplers...')
    train_size = int(len(state_space_samplers) * 0.8)
    train_state_space_samplers = state_space_samplers[:train_size]
    validation_state_space_samplers = state_space_samplers[train_size:]
    train_dataset = StateDataset(train_state_space_samplers)
    validation_dataset = StateDataset(validation_state_space_samplers)
    return train_dataset, validation_dataset


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int, aggregation: str, device: torch.device) -> nn.Module:
    return RelationalGraphNeuralNetwork(
        domain=domain,
        aggregation=aggregation,
        embedding_size=embedding_size,
        num_layers=num_layers,
        global_readout=False,
        normalization=True,
        random_initialization=False,
    ).to(device)


def _sample_batch(state_sampler: StateDataset, batch_size: int, device: torch.device) -> tuple[list[mm.State], torch.Tensor]:
    states: list[mm.State] = []
    targets: list[float]  = []
    for _ in range(batch_size):
        state, label = state_sampler.sample()
        states.append(state)
        targets.append(1000.0 if label.is_dead_end else label.steps_to_goal)
    return states, torch.tensor(targets,  dtype=torch.float, requires_grad=False, device=device)


def _train(model: RelationalGraphNeuralNetwork,
           optimizer: optim.Adam,
           train_states: StateDataset,
           validation_states: StateDataset,
           num_epochs: int,
           batch_size: int,
           device: torch.device) -> None:
    TRAIN_SIZE = 1000
    VALIDATION_SIZE = 100
    # Training loop
    best_error = None  # Track the best validation loss to detect overfitting.
    print('Training model...')
    for epoch in range(0, num_epochs):
        # Train step
        for index in range(TRAIN_SIZE):
            states, targets = _sample_batch(train_states, batch_size, device)
            predictions = model.forward_value(states)
            loss = (predictions - targets).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print loss every 100 steps (printing every step forces synchronization with CPU)
            if (index + 1) % 100 == 0:
                print(f'[{epoch + 1}/{num_epochs}; {index + 1}/{TRAIN_SIZE}] Loss: {loss.item():.4f}')
        # Validation step
        with torch.no_grad():
            error = torch.zeros([1], dtype=torch.float, device=device)
            for index in range(VALIDATION_SIZE):
                states, targets = _sample_batch(validation_states, batch_size, device)
                predictions = model.forward_value(states)
                error += (predictions - targets).abs().sum()
            total_samples = VALIDATION_SIZE * batch_size
            error = error / total_samples
            print(f'[{epoch + 1}/{num_epochs}] Absolute error: {error.item():.4f}')
            model.save('latest.pth', { 'optimizer': optimizer.state_dict() })
            if (best_error is None) or (error < best_error):
                best_error = error
                model.save('best.pth', { 'optimizer': optimizer.state_dict() })
                print(f'[{epoch + 1}/{num_epochs}] Saved new best model')


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    device = create_device()
    domain_path, problem_paths = _get_instance_paths(args.input)
    domain, problems = _parse_instances(domain_path, problem_paths)
    state_spaces = _generate_state_spaces(problems, args.seed)
    train_dataset, validation_dataset = _create_datasets(state_spaces)
    if args.model is None:
        print('Creating a new model and optimizer...')
        model = _create_model(domain, args.embedding_size, args.layers, args.aggregation, device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        print(f'Loading an existing model and optimizer... ({args.model})')
        model, extras = RelationalGraphNeuralNetwork.load(domain, args.model, device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(extras['optimizer'])
    _train(model, optimizer, train_dataset, validation_dataset, args.num_epochs, args.batch_size, device)


if __name__ == "__main__":
    _main(_parse_arguments())
