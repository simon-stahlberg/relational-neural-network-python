import pymimir as mm
import torch

from typing import Union


class StateWrapper:
    def __init__(self, state: mm.State, problem: mm.Problem, repositories: mm.PDDLRepositories):
        self.state = state
        self.problem = problem
        self.repositories = repositories


def get_atom_name(atom: Union[mm.StaticAtom, mm.FluentAtom, mm.DerivedAtom], is_goal_atom: bool):
    if is_goal_atom: return get_predicate_name(atom.get_predicate(), True)
    else: return get_predicate_name(atom.get_predicate(), False)


def get_predicate_name(predicate: Union[mm.StaticPredicate, mm.FluentPredicate, mm.DerivedPredicate], is_goal_predicate: bool):
    if is_goal_predicate: return ('relation_' + predicate.get_name() + '_goal')
    else: return 'relation_' + predicate.get_name()


def get_state_atoms(state_wrapper: StateWrapper):
    atoms = list(state_wrapper.problem.get_static_initial_atoms())
    atoms.extend(state_wrapper.repositories.get_fluent_ground_atoms_from_indices(state_wrapper.state.get_fluent_atoms()))
    atoms.extend(state_wrapper.repositories.get_derived_ground_atoms_from_indices(state_wrapper.state.get_derived_atoms()))
    return atoms


def get_goal_atoms(state_wrapper: StateWrapper):
    atoms = [literal.get_atom() for literal in state_wrapper.problem.get_static_goal_condition()]
    atoms.extend([literal.get_atom() for literal in state_wrapper.problem.get_fluent_goal_condition()])
    atoms.extend([literal.get_atom() for literal in state_wrapper.problem.get_derived_goal_condition()])
    return atoms


def create_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU: ", torch.cuda.get_device_name(0))
    # The MPS implementation does not yet support all operations that we use.
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("MPS is available. Using MPS.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device


def relations_to_tensors(term_id_groups: dict[str, list[int]], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result
