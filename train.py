import torch

import argparse
import pymimir
import random

from pathlib import Path
from typing import List, Tuple


class StateSpaceDataset:
    def __init__(self, state_spaces: List[pymimir.StateSpace], num_items: int) -> None:
        self._state_spaces = state_spaces
        self._num_items = num_items

    def __len__(self):
        # We assume that the size of the dataset is 'self._num_items'.
        # This ensures predictable epoch lengths, regardless of the size of the given state spaces.
        # In the getitem method, we randomly sample states instead of systematically enumerating them.
        return self._num_items

    def __getitem__(self, ignored_index) -> Tuple[pymimir.State, pymimir.StateSpace, int]:
        # We ignore the index and return a random state.
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
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per batch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate for the training process')
    args = parser.parse_args()
    return args


def _parse_instances(input: Path):
    print('Parsing files...')
    if input.is_file():
        domain_file = str(input.parent / 'domain.pddl')
        problem_files = [str(input)]
    else:
        domain_file = str(input / 'domain.pddl')
        problem_files = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_files.sort()
    return domain_file, problem_files


def _generate_state_spaces(domain_path: str, problem_paths: List[str]):
    print('Generating state spaces...')
    state_spaces: List[pymimir.StateSpace] = []
    for problem_path in problem_paths:
        print(f'> Expanding: {problem_path}')
        state_space = pymimir.StateSpace.create(domain_path, problem_path, 1_000_000, 60_000)
        if state_space is not None:
            state_spaces.append(state_space)
            print(f'- # States: {state_space.get_num_states()}')
        else:
            print('- Skipped')
    state_spaces.sort(key=lambda state_space: state_space.get_num_states())
    return state_spaces


def _create_datasets(state_spaces: List[pymimir.StateSpace]):
    print('Create datasets...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSpaceDataset(train_state_spaces, 10_000)
    validation_dataset = StateSpaceDataset(validation_state_spaces, 1_000)
    return train_dataset, validation_dataset


def _train(train_dataset: pymimir.StateSpace, validation_dataset: pymimir.StateSpace):
    pass


def _main(args: argparse.Namespace):
    domain_path, problem_paths = _parse_instances(args.input)
    state_spaces = _generate_state_spaces(domain_path, problem_paths)
    train_dataset, validation_dataset = _create_datasets(state_spaces)
    _train(train_dataset, validation_dataset)


if __name__ == "__main__":
    _main(_parse_arguments())
