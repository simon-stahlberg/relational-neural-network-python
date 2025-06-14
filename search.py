import argparse
import pymimir as mm
import torch

from pathlib import Path
from rgnn import RelationalGraphNeuralNetwork
from utils import create_device


class NeuralHeuristic(mm.Heuristic):
    def __init__(self, problem: mm.Problem, model: RelationalGraphNeuralNetwork):
        super().__init__()
        self._problem = problem
        self._model = model

    def compute_value(self, state: mm.State, is_goal_state: bool) -> float:
        if is_goal_state: return 0.0
        with torch.no_grad():
            self._model.eval()
            value = self._model.forward_value([state])[0]
            return value


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for testing')
    parser.add_argument('--domain', required=True, type=Path, help='Path to the domain file')
    parser.add_argument('--problem', required=True, type=Path, help='Path to the problem file')
    parser.add_argument('--model', required=True, type=Path, help='Path to a pre-trained model')
    args = parser.parse_args()
    return args


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    domain = mm.Domain(args.domain)
    problem = mm.Problem(domain, args.problem)
    print(f'Loading model... ({args.model})')
    device = create_device()
    model, _ = RelationalGraphNeuralNetwork.load(domain, args.model, device)
    initial_state = problem.get_initial_state()
    neural_heuristic = NeuralHeuristic(problem, model)
    # Initialize counters for statistics.
    num_expanded = 0
    num_generated = 0
    def increment_expanded(state):
        nonlocal num_expanded
        num_expanded += 1
    def increment_generated(state, action, cost, successor_state):
        nonlocal num_generated
        num_generated += 1
    def print_f_layer(f: float):
        print(f'[f={f:.3f}] Expanded: {num_expanded}, Generated: {num_generated}')
    # Start the A* search with eager evaluation.
    result = mm.astar_eager(
        problem,
        initial_state,
        neural_heuristic,
        on_expand_state=increment_expanded,
        on_generate_state=increment_generated,
        on_finish_f_layer=print_f_layer,
    )
    # Print the statistics.
    print(f'[Final] Expanded: {num_expanded}, Generated: {num_generated}')
    # Print the result of the search.
    if result.status == "solved":
        print(f'Found a solution of length {len(result.solution)}!')
        for index, action in enumerate(result.solution):
            print(f'{index + 1:>3}: {str(action)}')
    else:
        print('Failed to find a solution!')


if __name__ == '__main__':
    _main(_parse_arguments())
