import argparse
import pymimir as mm
import torch

from pathlib import Path
from typing import List, Union
from rgnn import RelationalGraphNeuralNetwork
from utils import create_device


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for testing')
    parser.add_argument('--domain', required=True, type=Path, help='Path to the domain file')
    parser.add_argument('--problem', required=True, type=Path, help='Path to the problem file')
    parser.add_argument('--model', required=True, type=Path, help='Path to a pre-trained model')
    args = parser.parse_args()
    return args


def _plan(problem: mm.Problem, model: RelationalGraphNeuralNetwork) -> Union[None, List[mm.GroundAction]]:
    # Disable gradient as we are not optimizing.
    with torch.no_grad():
        solution = []
        goal = problem.get_goal_condition()
        current_state = problem.get_initial_state()
        while not goal.holds(current_state) and (len(solution) < 1_000):
            applicable_actions = current_state.generate_applicable_actions()
            successor_states = [action.apply(current_state) for action in applicable_actions]
            values = model.forward_value(successor_states)
            assert isinstance(values, torch.Tensor), 'Model should return a tensor of values.'
            values = values.cpu()  # Move the result to the CPU.
            min_index = values.argmin().item()
            min_value = values[min_index].item()
            selected_action = applicable_actions[min_index]
            selected_successor = successor_states[min_index]
            current_state = selected_successor
            solution.append(selected_action)
            print(f'{min_value:.3f}: {str(selected_action)}')
        return solution if goal.holds(current_state) else None


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    domain = mm.Domain(args.domain)
    problem = mm.Problem(domain, args.problem)
    model, _ = RelationalGraphNeuralNetwork.load(domain, args.model, create_device())
    solution = _plan(problem, model)
    if solution is None:
        print('Failed to find a solution!')
    else:
        print(f'Found a solution of length {len(solution)}!')
        for index, action in enumerate(solution):
            print(f'{index + 1:>4}: {str(action)}')


if __name__ == '__main__':
    _main(_parse_arguments())
