import pymimir as mm
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import mish

from utils import StateWrapper, get_atom_name, get_goal_atoms, get_predicate_name, get_state_atoms, relations_to_tensors


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._inner = nn.Linear(input_size, input_size, True)
        self._outer = nn.Linear(input_size, output_size, True)

    def forward(self, input):
        return self._outer(mish(self._inner(input)))


class SumReadout(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._value = MLP(input_size, output_size)

    def forward(self, node_embeddings: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
        cumsum_indices = node_sizes.cumsum(0) - 1
        cumsum_states = node_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)


class RelationMessagePassingBase(nn.Module):
    def __init__(self, predicate_signatures: list[tuple[str, int]], embedding_size: int):
        super().__init__()
        predicate_signatures.sort()  # Ensure that relations are always processed in the same order
        self._embedding_size = embedding_size
        self._relation_mlps = nn.ModuleDict()
        for relation_name, relation_arity in predicate_signatures:
            input_size = relation_arity * embedding_size
            output_size = relation_arity * embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)

    def _compute_messages_and_indices(self, node_embeddings: torch.Tensor, atoms: dict[str, torch.Tensor]):
        output_messages_list = []
        output_indices_list = []
        for relation_name, relation_module in self._relation_mlps.items():
            if relation_name in atoms:
                atom_values = atoms[relation_name]
                input_embeddings = torch.index_select(node_embeddings, 0, atom_values).view(-1, relation_module.input_size)
                output_messages = (input_embeddings + relation_module(input_embeddings)).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(atom_values)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        return output_messages, output_indices


class MeanRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, predicate_signatures: list[tuple[str, int]], embedding_size: int):
        super().__init__(predicate_signatures, embedding_size)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, atoms)
        count_messages = torch.ones_like(output_messages)
        sum_msg = torch.zeros_like(node_embeddings)
        cnt_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        cnt_msg.index_add_(0, output_indices, count_messages)
        avg_msg = sum_msg / (cnt_msg + 1E-16)
        return self._update_mlp(torch.cat((avg_msg, node_embeddings), 1))


class SumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, atoms)
        sum_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        return self._update_mlp(torch.cat((sum_msg, node_embeddings), 1))


class SmoothMaximumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, atoms)
        # To achieve numerical stability, shift using the largest values.
        with torch.no_grad():
            exps_max = torch.zeros_like(node_embeddings)
            exps_max.index_reduce_(0, output_indices, output_messages, reduce="amax", include_self=False)
            max_offsets = exps_max.index_select(0, output_indices)
        # Use smooth-maximum instead of hard maximum to ensure that all senders get some gradient.
        MAXIMUM_SMOOTHNESS = 12.0  # As the value approaches infinity, the hard maximum is attained
        exps = (MAXIMUM_SMOOTHNESS * (output_messages - max_offsets)).exp()
        exps_sum = torch.full_like(node_embeddings, 1E-16)
        exps_sum.index_add_(0, output_indices, exps)
        max_msg = ((1.0 / MAXIMUM_SMOOTHNESS) * exps_sum.log()) + exps_max
        return self._update_mlp(torch.cat((max_msg, node_embeddings), 1))


class RelationalMessagePassingModule(nn.Module):
    def __init__(self, predicate_signatures: list[tuple[str, int]], embedding_size: int, num_layers: int, aggregation: str = 'max'):
        super().__init__()
        self._num_layers = num_layers
        self._embedding_size = embedding_size
        if aggregation == 'max': self._relation_network = SmoothMaximumRelationMessagePassing(predicate_signatures, embedding_size)
        elif aggregation == 'sum': self._relation_network = SumRelationMessagePassing(predicate_signatures, embedding_size)
        elif aggregation == 'mean': self._relation_network = MeanRelationMessagePassing(predicate_signatures, embedding_size)
        else: raise ValueError(f'invalid aggregation: "{aggregation}"')

    def forward(self, node_embeddings: torch.Tensor, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
        for _ in range(self._num_layers):
            node_embeddings = node_embeddings + self._relation_network(node_embeddings, atoms)
        return node_embeddings


class StateEncoder(nn.Module):
    def __init__(self, domain: mm.Domain, embedding_size: int):
        super().__init__()
        self._domain = domain
        self._embedding_size = embedding_size
        self._dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy.device

    def get_predicate_signatures(self) -> list[tuple[str, int]]:
        predicates = []
        predicates.extend(self._domain.get_static_predicates())
        predicates.extend(self._domain.get_fluent_predicates())
        predicates.extend(self._domain.get_derived_predicates())
        relation_name_arities = [(get_predicate_name(predicate, False), len(predicate.get_parameters())) for predicate in predicates]
        relation_name_arities.extend([(get_predicate_name(predicate, True), len(predicate.get_parameters())) for predicate in predicates])
        return relation_name_arities

    def forward(self, state_wrappers: list[StateWrapper]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]]:
        device = self.get_device()
        atoms = {}
        token_sizes = []
        offset = 0
        # Helper function for populating relations and sizes.
        def add_atom_relation(atom, is_goal_atom):
            relation_name = get_atom_name(atom, is_goal_atom)
            term_ids = [term.get_index() + offset for term in atom.get_objects()]
            # Add atoms
            if relation_name not in atoms: atoms[relation_name] = term_ids
            else: atoms[relation_name].extend(term_ids)
        # Construct input
        for state_wrapper in state_wrappers:
            objects = state_wrapper.problem.get_objects()
            # Add state relations
            for atom in get_state_atoms(state_wrapper):
                add_atom_relation(atom, False)
            # Add goal relations
            for atom in get_goal_atoms(state_wrapper):
                add_atom_relation(atom, True)
            # Situation sizes
            token_sizes.append(len(objects))
            offset += len(objects)
        # Move input to device
        node_embeddings = torch.zeros((offset, self._embedding_size), dtype=torch.float, requires_grad=True, device=device)
        atoms = relations_to_tensors(atoms, device)
        token_sizes = torch.tensor(token_sizes, dtype=torch.int, device=device, requires_grad=False)
        return node_embeddings, atoms, token_sizes


class RelationalGraphNeuralNetwork(nn.Module):
    def __init__(self, domain: mm.Domain, embedding_size: int, num_layers: int, aggregation: str = 'max'):
        super().__init__()
        self._encoder = StateEncoder(domain, embedding_size)
        self._domain = domain
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._aggregation = aggregation
        self._mpnn_module = RelationalMessagePassingModule(self._encoder.get_predicate_signatures(), embedding_size, num_layers, aggregation)
        self._value_readout = SumReadout(embedding_size, 1)
        self._deadend_readout = SumReadout(embedding_size, 1)

    def forward(self, states: list[StateWrapper]) -> tuple[torch.Tensor, torch.Tensor]:
        node_embeddings, atoms, token_sizes, _ = self._encoder(states)
        node_embeddings = self._mpnn_module(node_embeddings, atoms)
        values = self._value_readout(node_embeddings, token_sizes).view(-1)
        deadends = self._deadend_readout(node_embeddings, token_sizes).view(-1)
        return values, deadends

    def get_state_and_hparams_dicts(self):
        return self.state_dict(), { 'embedding_size': self._embedding_size, 'num_layers': self._num_layers, 'aggregation': self._aggregation }


def save_checkpoint(model: RelationalGraphNeuralNetwork, optimizer: optim.Adam, path: str) -> None:
    model_dict, hparams_dict = model.get_state_and_hparams_dicts()
    checkpoint = { 'model': model_dict, 'hparams': hparams_dict, 'optimizer': optimizer.state_dict() }
    torch.save(checkpoint, path)


def load_checkpoint(domain: mm.Domain, path: str, device: torch.device) -> tuple[RelationalGraphNeuralNetwork, optim.Adam]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hparams_dict = checkpoint['hparams']
    model = RelationalGraphNeuralNetwork(domain, **hparams_dict)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
