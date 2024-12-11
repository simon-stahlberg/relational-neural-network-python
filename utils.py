import pymimir as mm
import torch
import torch.optim as optim

from relnn_max import SmoothmaxRelationalNeuralNetwork
from typing import Dict, List, Union


def get_atom_name(atom: Union[mm.StaticAtom, mm.FluentAtom, mm.DerivedAtom], state: mm.State, is_goal_atom: bool):
    if is_goal_atom: return get_predicate_name(atom.get_predicate(), True, state.contains(atom))
    else: return get_predicate_name(atom.get_predicate(), False, True)


def get_predicate_name(predicate: Union[mm.StaticPredicate, mm.FluentPredicate, mm.DerivedPredicate], is_goal_predicate: bool, is_true: bool):
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate: return ('relation_' + predicate.get_name() + '_goal') + ('_true' if is_true else '_false')
    else: return 'relation_' + predicate.get_name()


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


def save_checkpoint(model: SmoothmaxRelationalNeuralNetwork, optimizer: optim.Adam, path: str):
    model_dict, hparams_dict = model.get_state_and_hparams_dicts()
    checkpoint = { 'model': model_dict, 'hparams': hparams_dict, 'optimizer': optimizer.state_dict() }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hparams_dict = checkpoint['hparams']
    model = SmoothmaxRelationalNeuralNetwork(hparams_dict['predicates'], hparams_dict['embedding_size'], hparams_dict['num_layers'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def get_atoms(state: mm.State, problem: mm.Problem, factories: mm.PDDLRepositories) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    atoms = [literal.get_atom() for literal in problem.get_static_initial_literals()]
    atoms.extend(factories.get_fluent_ground_atoms_from_indices(state.get_fluent_atoms()))
    atoms.extend(factories.get_derived_ground_atoms_from_indices(state.get_derived_atoms()))
    return atoms


def get_goal(problem: mm.Problem) -> List[Union[mm.StaticGroundAtom, mm.FluentGroundAtom, mm.DerivedGroundAtom]]:
    static_goal = [literal.get_atom() for literal in problem.get_static_goal_condition()]
    fluent_goal = [literal.get_atom() for literal in problem.get_fluent_goal_condition()]
    derived_goal = [literal.get_atom() for literal in problem.get_derived_goal_condition()]
    full_goal = static_goal + fluent_goal + derived_goal
    return full_goal


def relations_to_tensors(term_id_groups: Dict[str, List[int]], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


def create_input(problem: mm.Problem, states: List[mm.State], factories: mm.PDDLRepositories, device: torch.device):
    relations = {}
    sizes = []
    # Helper function for populating relations and sizes.
    def add_relations(atom, offset, is_goal_atom):
        predicate_name = get_atom_name(atom, state, is_goal_atom)
        term_ids = [term.get_index() + offset for term in atom.get_objects()]
        if predicate_name not in relations: relations[predicate_name] = term_ids
        else: relations[predicate_name].extend(term_ids)
    # Add all states to relations and sizes, together with the goal.
    for state in states:
        offset = sum(sizes)
        for atom in get_atoms(state, problem, factories): add_relations(atom, offset, False)
        for atom in get_goal(problem): add_relations(atom, offset, True)
        sizes.append(len(problem.get_objects()))
    # Move all lists to the GPU as tensors.
    return relations_to_tensors(relations, device), torch.tensor(sizes, dtype=torch.int, device=device, requires_grad=False)
