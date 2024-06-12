import torch
import torch.nn as nn

from typing import List, Dict, Tuple
from torch.nn.functional import mish


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._inner = nn.Linear(input_size, input_size, True)
        self._outer = nn.Linear(input_size, output_size, True)

    def forward(self, input):
        return self._outer(mish(self._inner(input)))


class RelationMessagePassing(nn.Module):
    def __init__(self, relation_name_arities: List[Tuple[str, int]], embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self._relation_mlps = nn.ModuleDict()
        for relation_name, relation_arity in relation_name_arities:
            input_size = relation_arity * embedding_size
            output_size = relation_arity * embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)

    def forward(self, object_embeddings: torch.Tensor, relations: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_messages_list = []
        output_indices_list = []
        for relation_name, relation_module in self._relation_mlps.items():
            if relation_name in relations:
                relation_values = relations[relation_name]
                input_embeddings = torch.index_select(object_embeddings, 0, relation_values).view(-1, relation_module.input_size)
                output_messages = (input_embeddings + relation_module(input_embeddings)).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(relation_values)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        exps_max = torch.zeros_like(object_embeddings)
        exps_max.index_reduce_(0, output_indices, output_messages, reduce="amax", include_self=False)
        exps_max = exps_max.detach()
        MAXIMUM_SMOOTHNESS = 12.0  # As the value approaches infinity, the hard maximum is attained
        max_offsets = exps_max.index_select(0, output_indices).detach()
        exps = (MAXIMUM_SMOOTHNESS * (output_messages - max_offsets)).exp()
        exps_sum = torch.full_like(object_embeddings, 1E-16)
        exps_sum.index_add_(0, output_indices, exps)
        max_msg = ((1.0 / MAXIMUM_SMOOTHNESS) * exps_sum.log()) + exps_max
        next_object_embeddings = object_embeddings + self._update_mlp(torch.cat([max_msg, object_embeddings], 1))
        return next_object_embeddings


class SumReadout(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._value = MLP(input_size, output_size)

    def forward(self, object_embeddings: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        cumsum_indices = sizes.cumsum(0) - 1
        cumsum_states = object_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)


class RelationalMessagePassingModule(nn.Module):
    def __init__(self, relation_name_arities: List[Tuple[str, int]], embedding_size: int, num_layers: int):
        super().__init__()
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._relation_network = RelationMessagePassing(relation_name_arities, embedding_size)
        self._dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy.device

    def forward(self, relations: Dict[str, torch.Tensor], sizes: torch.Tensor) -> torch.Tensor:
        object_embeddings = self._initialize_nodes(sum(sizes))
        object_embeddings = self._pass_messages(object_embeddings, relations)
        return object_embeddings

    def _pass_messages(self, object_embeddings: torch.Tensor, relations: Dict[str, torch.Tensor]) -> torch.Tensor:
        for _ in range(self._num_layers):
            object_embeddings = self._relation_network(object_embeddings, relations)
        return object_embeddings

    def _initialize_nodes(self, num_objects: int) -> torch.Tensor:
        object_embeddings = torch.zeros((num_objects, self._embedding_size), dtype=torch.float, device=self.get_device())
        return object_embeddings


class SmoothmaxRelationalNeuralNetwork(nn.Module):
    def __init__(self, predicates: List[Tuple[str, int]], embedding_size: int, num_layers: int):
        super().__init__()
        predicates.sort()  # Ensure that relations are always processed in the same order
        self._predicates = predicates
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._module = RelationalMessagePassingModule(predicates, embedding_size, num_layers)
        self._readout = SumReadout(embedding_size, 1)

    def forward(self, relations: Dict[str, torch.Tensor], sizes: torch.Tensor) -> torch.Tensor:
        object_embeddings = self._module(relations, sizes)
        return self._readout(object_embeddings, sizes)

    def get_state_and_hparams_dicts(self):
        return self.state_dict(), { 'predicates': self._predicates, 'embedding_size': self._embedding_size, 'num_layers': self._num_layers }
