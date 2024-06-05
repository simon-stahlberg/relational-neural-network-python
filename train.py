import argparse
import pymimir as mm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from pathlib import Path
from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import Dict, List, Tuple, Union


class StateSpaceDataset:
    def __init__(self, state_spaces: List[mm.StateSpace], num_items: int) -> None:
        self._state_spaces = state_spaces
        self._num_items = num_items

    def __len__(self) -> int:
        # We assume that the size of the dataset is 'self._num_items'.
        # This ensures predictable epoch lengths, regardless of the size of the given state spaces.
        # In the getitem method, we randomly sample states instead of systematically enumerating them.
        return self._num_items

    def __getitem__(self, ignored_index) -> Tuple[mm.State, mm.StateSpace, int]:
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


def _create_datasets(state_spaces: List[mm.StateSpace]) -> Tuple[StateSpaceDataset, StateSpaceDataset]:
    print('Create datasets...')
    train_size = int(len(state_spaces) * 0.8)
    train_state_spaces = state_spaces[:train_size]
    validation_state_spaces = state_spaces[train_size:]
    train_dataset = StateSpaceDataset(train_state_spaces, 10_000)
    validation_dataset = StateSpaceDataset(validation_state_spaces, 1_000)
    return train_dataset, validation_dataset


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int) -> nn.Module:
    predicates = domain.get_static_predicates() + domain.get_fluent_predicates() + domain.get_derived_predicates()
    relation_name_arities = [(predicate.get_name(), len(predicate.get_parameters())) for predicate in predicates]
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


def _group_term_ids_by_predicate_name(state_atoms: List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]],
                                      goal_atoms: List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for atom in state_atoms:
        predicate_name = atom.get_predicate().get_name()
        term_ids = [term.get_identifier() for term in atom.get_objects()]
        groups[predicate_name].extend(term_ids)
    for atom in goal_atoms:
        predicate_name = atom.get_predicate().get_name()
        term_ids = [term.get_identifier() for term in atom.get_objects()]
        groups[predicate_name + '_goal'].extend(term_ids)
    return groups


def _term_id_groups_to_tensors(term_id_groups: Dict[str, List[int]], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


def _add_value_to_term_ids_in_group(group: Dict[str, List[int]], value: int) -> None:
    for terms in group.values():
        for index in range(len(terms)):
            terms[index] += value


def _create_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device


def _save_checkpoint(model: SmoothmaxRelationalNeuralNetwork, optimizer: optim.Adam, path: str):
    model_dict, hparams_dict = model.get_state_and_hparams_dicts()
    checkpoint = { 'model': model_dict, 'hparams': hparams_dict, 'optimizer': optimizer.state_dict() }
    torch.save(checkpoint, path)


def _load_checkpoint(path: str):
    checkpoint = torch.load(path)
    hparams_dict = checkpoint['hparams']
    model = SmoothmaxRelationalNeuralNetwork(hparams_dict['predicates'], hparams_dict['embedding_size'], hparams_dict['num_layers'])
    model.load_state_dict(checkpoint['model'])
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def _train(model: SmoothmaxRelationalNeuralNetwork, train_dataset: StateSpaceDataset, validation_dataset: StateSpaceDataset, num_epochs: int) -> None:
    device = _create_device()
    model = model.to(device)
    state, state_space, goal_distance = train_dataset[0]
    num_objects = len(state_space.get_problem().get_objects())
    state_atoms = _get_atoms(state, state_space)
    goal_atoms = _get_goal(state_space)
    term_id_groups = _group_term_ids_by_predicate_name(state_atoms, goal_atoms)
    input = _term_id_groups_to_tensors(term_id_groups, device)
    # Training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    for epoch in range(0, num_epochs):
        # Forward pass
        output = model(input, [num_objects]).view(-1)
        target = torch.tensor([goal_distance], dtype=torch.float, device=device)
        loss: torch.Tensor = criterion(output, target)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0: print(f'[{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}')
    pass


def _main(args: argparse.Namespace) -> None:
    domain_path, problem_paths = _parse_instances(args.input)
    state_spaces = _generate_state_spaces(domain_path, problem_paths)
    train_dataset, validation_dataset = _create_datasets(state_spaces)
    domain = state_spaces[0].get_problem().get_domain()
    model = _create_model(domain, args.embedding_size, args.layers)
    _train(model, train_dataset, validation_dataset, args.num_epochs)


if __name__ == "__main__":
    _main(_parse_arguments())
