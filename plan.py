import argparse
import pymimir as mm
import torch

from pathlib import Path
from typing import List, Union
from rgnn import RelationalGraphNeuralNetwork, load_checkpoint
from utils import StateWrapper, create_device


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for testing')
    parser.add_argument('--domain', required=True, type=Path, help='Path to the domain file')
    parser.add_argument('--problem', required=True, type=Path, help='Path to the problem file')
    parser.add_argument('--model', required=True, type=Path, help='Path to a pre-trained model')
    args = parser.parse_args()
    return args


def _plan(problem: mm.Problem, repositories: mm.PDDLRepositories, model: RelationalGraphNeuralNetwork) -> Union[None, List[mm.GroundAction]]:
    solution = []
    # Helper function for testing is a state is a goal state.
    def is_goal_state(state: mm.State) -> bool:
        return state.literals_hold(problem.get_fluent_goal_condition()) and state.literals_hold(problem.get_derived_goal_condition())
    # Disable gradient as we are not optimizing.
    with torch.no_grad():
        solution = []
        grounder = mm.Grounder(problem, repositories)
        aag_workspace = mm.ApplicableActionGeneratorWorkspace()
        successor_generator = mm.LiftedApplicableActionGenerator(grounder.get_action_grounder())
        axiom_evaluator = mm.LiftedAxiomEvaluator(grounder.get_axiom_grounder())
        state_repository = mm.StateRepository(axiom_evaluator)
        sr_workspace = mm.StateRepositoryWorkspace()
        current_state = state_repository.get_or_create_initial_state(sr_workspace)
        while (not is_goal_state(current_state)) and (len(solution) < 1_000):
            applicable_actions = successor_generator.generate_applicable_actions(current_state, aag_workspace)
            successor_states = [StateWrapper(state_repository.get_or_create_successor_state(current_state, action, sr_workspace)[0], problem, repositories) for action in applicable_actions]
            values, deadends = model.forward(successor_states)
            min_index = values.argmin()
            min_value = values[min_index]
            min_action = applicable_actions[min_index]
            min_successor = successor_states[min_index]
            current_state = min_successor.state
            solution.append(min_action)
            print(f'{min_value.item():.3f}: {min_action.to_string_for_plan(repositories)}')
        return solution if is_goal_state(current_state) else None


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}')
    parser = mm.PDDLParser(str(args.domain), str(args.problem))
    problem = parser.get_problem()
    repositories = parser.get_pddl_repositories()
    model, _ = load_checkpoint(problem.get_domain(), args.model, create_device())
    solution = _plan(problem, repositories, model)
    if solution is None:
        print('Failed to find a solution!')
    else:
        print(f'Found a solution of length {len(solution)}!')
        for index, action in enumerate(solution):
            print(f'{index + 1}: {str(action.to_string_for_plan(repositories))}')


if __name__ == '__main__':
    _main(_parse_arguments())
