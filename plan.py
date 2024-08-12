import argparse
import pymimir as mm
import torch

from pathlib import Path
from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import List, Union
from utils import create_device, load_checkpoint, create_input


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for testing')
    parser.add_argument('--input', required=True, type=Path, help='Path to the problem file')
    parser.add_argument('--model', required=True, type=Path, help='Path to a pre-trained model')
    args = parser.parse_args()
    return args


def _create_parser(input: Path) -> mm.PDDLParser:
    print('Creating parser...')
    if input.is_file():
        domain_file = str(input.parent / 'domain.pddl')
        problem_file = str(input)
    else:
        raise Exception('input is not a file')
    return mm.PDDLParser(domain_file, problem_file)


def _plan(problem: mm.Problem, factories: mm.PDDLFactories, model: SmoothmaxRelationalNeuralNetwork, device: torch.device) -> Union[None, List[str]]:
    solution = []
    # Helper function for testing is a state is a goal state.
    def is_goal_state(state: mm.State) -> bool:
        return state.literals_hold(problem.get_fluent_goal_condition()) and state.literals_hold(problem.get_derived_goal_condition())
    # Disable gradient as we are not optimizing.
    with torch.no_grad():
        successor_generator = mm.LiftedApplicableActionGenerator(problem, factories)
        state_repository = mm.StateRepository(successor_generator)
        current_state = state_repository.get_or_create_initial_state()
        while (not is_goal_state(current_state)) and (len(solution) < 1_000):
            applicable_actions = successor_generator.compute_applicable_actions(current_state)
            successor_states = [state_repository.get_or_create_successor_state(current_state, action) for action in applicable_actions]
            relations, sizes = create_input(problem, successor_states, factories, device)
            output = model.forward(relations, sizes).view(-1)
            min_index = output.argmin()
            min_value = output[min_index]
            min_action = applicable_actions[min_index]
            min_successor = successor_states[min_index]
            current_state = min_successor
            solution.append(str(min_action))
            print(f'{min_value.item():.3f}: {min_action}')
    return solution if is_goal_state(current_state) else None


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    device = create_device()
    parser = _create_parser(args.input)
    print(f'Loading model... ({args.model})')
    model, _ = load_checkpoint(args.model, device)
    solution = _plan(parser.get_problem(), parser.get_pddl_factories(), model, device)
    if solution is None:
        print('Failed to find a solution!')
    else:
        print(f'Found a solution of length {len(solution)}!')
        for index, action in enumerate(solution):
            print(f'{index + 1}: {str(action)}')


if __name__ == '__main__':
    _main(_parse_arguments())
