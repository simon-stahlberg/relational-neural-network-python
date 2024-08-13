import argparse
import pymimir as mm
import torch

from pathlib import Path
from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import List
from utils import create_device, load_checkpoint, create_input


class NeuralHeuristic(mm.IHeuristic):
    def __init__(self, problem: mm.Problem, pddl_factories: mm.PDDLFactories, model: SmoothmaxRelationalNeuralNetwork, device: torch.device):
        mm.IHeuristic.__init__(self)
        self._problem = problem
        self._pddl_factories = pddl_factories
        self._model = model
        self._device = device

    def compute_heuristic(self, state: mm.State) -> float:
        relations, sizes = create_input(self._problem, [state], self._pddl_factories, self._device)
        output = self._model.forward(relations, sizes).view(-1)
        return output[0].item()


class AStarEventHandler(mm.AStarAlgorithmEventHandlerBase):
    def __init__(self, quiet = True):
        mm.AStarAlgorithmEventHandlerBase.__init__(self, quiet)

    def on_expand_state_impl(self, state: mm.State, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_generate_state_impl(self, state: mm.State, action: mm.GroundAction, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_generate_state_relaxed_impl(self, state: mm.State, action: mm.GroundAction, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_generate_state_not_relaxed_impl(self, state: mm.State, action: mm.GroundAction, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_close_state_impl(self, state: mm.State, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_finish_f_layer_impl(self, f_value: float, num_expanded_states: int, num_generated_states: int): print(f'[{f_value:.3f}] Expanded: {num_expanded_states}; Generated: {num_generated_states}')
    def on_prune_state_impl(self, state: mm.State, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_start_search_impl(self, start_state: mm.State, problem: mm.Problem, pddl_factories: mm.PDDLFactories): pass
    def on_end_search_impl(self): print('Search finished')
    def on_solved_impl(self, ground_action_plan: List[mm.GroundAction]): pass
    def on_unsolvable_impl(self): pass
    def on_exhausted_impl(self): pass


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


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    device = create_device()
    parser = _create_parser(args.input)
    print(f'Loading model... ({args.model})')
    model, _ = load_checkpoint(args.model, device)
    problem = parser.get_problem()
    pddl_factories = parser.get_pddl_factories()
    lifted_aag = mm.LiftedApplicableActionGenerator(problem, pddl_factories)
    state_repository = mm.StateRepository(lifted_aag)
    neural_heuristic = NeuralHeuristic(problem, pddl_factories, model, device)
    event_handler = AStarEventHandler(False)
    astar_search_algorithm = mm.AStarAlgorithm(lifted_aag, state_repository, neural_heuristic, event_handler)
    search_status, plan = astar_search_algorithm.find_solution()
    if search_status == mm.SearchStatus.SOLVED:
        print(f'Solved using {len(plan)} actions')
        for index, action in enumerate(plan):
            print(f'{index + 1}: {str(action)}')
    else:
        print('Failed solving problem')


if __name__ == '__main__':
    _main(_parse_arguments())
