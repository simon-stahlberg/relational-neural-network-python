import argparse
import pymimir as mm
import torch

from pathlib import Path
from typing import List
from rgnn import RelationalGraphNeuralNetwork, load_checkpoint
from utils import StateWrapper, create_device


class NeuralHeuristic(mm.IHeuristic):
    def __init__(self, problem: mm.Problem, pddl_repositories: mm.PDDLRepositories, model: RelationalGraphNeuralNetwork, device: torch.device):
        mm.IHeuristic.__init__(self)
        self._problem = problem
        self._pddl_repositories = pddl_repositories
        self._model = model
        self._device = device

    def compute_heuristic(self, state: mm.State, is_goal: bool) -> float:
        if is_goal: return 0.0
        input = [StateWrapper(state, self._problem, self._pddl_repositories)]
        values, deadends = self._model.forward(input)
        # TODO: Take deadends into account.
        return values[0].item()


class AStarEventHandler(mm.AStarAlgorithmEventHandlerBase):
    def __init__(self, quiet = True):
        mm.AStarAlgorithmEventHandlerBase.__init__(self, quiet)
        self.expanded_states = 0
        self.generated_states = 0

    def on_expand_state_impl(self, state: mm.State, problem: mm.Problem, pddl_repositories: mm.PDDLRepositories): self.expanded_states += 1
    def on_generate_state_impl(self, state: mm.State, action: mm.GroundAction, cost: float, problem: mm.Problem, pddl_repositories: mm.PDDLRepositories): self.generated_states += 1
    def on_finish_f_layer_impl(self, f_value: float, num_expanded_states: int, num_generated_states: int): print(f'[f = {f_value:.3f}] Expanded: {self.expanded_states}; Generated: {self.generated_states}')


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for testing')
    parser.add_argument('--domain', required=True, type=Path, help='Path to the domain file')
    parser.add_argument('--problem', required=True, type=Path, help='Path to the problem file')
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
    parser = mm.PDDLParser(str(args.domain), str(args.problem))
    problem = parser.get_problem()
    domain = problem.get_domain()
    repositories = parser.get_pddl_repositories()
    print(f'Loading model... ({args.model})')
    device = create_device()
    model, _ = load_checkpoint(domain, args.model, device)
    grounder = mm.Grounder(problem, repositories)
    successor_generator = mm.LiftedApplicableActionGenerator(grounder.get_action_grounder())
    axiom_evaluator = mm.LiftedAxiomEvaluator(grounder.get_axiom_grounder())
    state_repository = mm.StateRepository(axiom_evaluator)
    sr_workspace = mm.StateRepositoryWorkspace()
    initial_state = state_repository.get_or_create_initial_state(sr_workspace)
    neural_heuristic = NeuralHeuristic(problem, repositories, model, device)
    event_handler = AStarEventHandler(False)
    search_result = mm.find_solution_astar(successor_generator, state_repository, neural_heuristic, initial_state, event_handler)
    print(f'[Final] Expanded: {event_handler.expanded_states}; Generated: {event_handler.generated_states}')
    if search_result.status == mm.SearchStatus.SOLVED:
        print(f'Found a solution with cost {search_result.plan.get_cost()}')
        for index, action in enumerate(search_result.plan.get_actions()):
            print(f'{index + 1}: {action.to_string_for_plan(repositories)}')
    else:
        print('Failed solving problem')


if __name__ == '__main__':
    _main(_parse_arguments())
