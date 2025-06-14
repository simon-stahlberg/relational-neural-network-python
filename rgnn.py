import pymimir as mm
import random
import torch
import torch.nn as nn

from pathlib import Path
from torch.nn.functional import mish
from typing import Union


def get_atom_name(atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool):
    if is_goal_atom:
        is_in_state = state.contains(atom)
        return get_predicate_name(atom.get_predicate(), True, is_in_state)
    else:
        return get_predicate_name(atom.get_predicate(), False, True)


def get_predicate_name(predicate: mm.Predicate, is_goal_predicate: bool, is_true: bool):
    assert (not is_goal_predicate and is_true) or (is_goal_predicate)
    if is_goal_predicate:
        suffix = '_true' if is_true else '_false'
        return f'relation_{predicate.get_name()}_goal_{suffix}'
    else:
        return f'relation_{predicate.get_name()}'


def get_action_name(action: Union[mm.Action, mm.GroundAction]):
    return 'action_' + str(action.get_index())


def relations_to_tensors(term_id_groups: dict[str, list[int]], device: torch.device) -> 'dict[str, torch.Tensor]':
    result = {}
    for key, value in term_id_groups.items():
        result[key] = torch.tensor(value, dtype=torch.int, device=device, requires_grad=False)
    return result


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

    def forward(self, node_embeddings: torch.Tensor, node_sizes: torch.Tensor) -> 'torch.Tensor':
        cumsum_indices = node_sizes.cumsum(0) - 1
        cumsum_states = node_embeddings.cumsum(0).index_select(0, cumsum_indices)
        aggregated_embeddings = torch.cat((cumsum_states[0].view(1, -1), cumsum_states[1:] - cumsum_states[0:-1]))
        return self._value(aggregated_embeddings)


class RelationMessagePassingBase(nn.Module):
    def __init__(self, domain: mm.Domain, embedding_size: int):
        super().__init__()
        predicates = domain.get_predicates()
        relation_name_arities = [(get_predicate_name(predicate, False, True), len(predicate.get_parameters())) for predicate in predicates]
        relation_name_arities.extend([(get_predicate_name(predicate, True, True), len(predicate.get_parameters())) for predicate in predicates])
        relation_name_arities.extend([(get_predicate_name(predicate, True, False), len(predicate.get_parameters())) for predicate in predicates])
        relation_name_arities.extend([(get_action_name(action), action.get_arity() + 1) for action in domain.get_actions()])
        relation_name_arities.sort()  # Ensure that relations are always processed in the same order
        self._embedding_size = embedding_size
        self._relation_mlps = nn.ModuleDict()
        for relation_name, relation_arity in relation_name_arities:
            input_size = relation_arity * embedding_size
            output_size = relation_arity * embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)
        self._update_mlp = MLP(2 * embedding_size, embedding_size)

    def _compute_messages_and_indices(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]):
        output_messages_list = []
        output_indices_list = []
        for relation_name, relation_module in self._relation_mlps.items():
            if relation_name in relations:
                relation_values = relations[relation_name]
                input_embeddings = torch.index_select(node_embeddings, 0, relation_values).view(-1, relation_module.input_size)
                output_messages = (input_embeddings + relation_module(input_embeddings)).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(relation_values)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        return output_messages, output_indices


class MeanRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        sum_msg = torch.zeros_like(node_embeddings)
        cnt_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        cnt_msg.index_add_(0, output_indices, torch.ones_like(output_messages))
        avg_msg = sum_msg / cnt_msg
        return self._update_mlp(torch.cat((avg_msg, node_embeddings), 1))


class SumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        sum_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        return self._update_mlp(torch.cat((sum_msg, node_embeddings), 1))


class HardMaximumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        max_msg = torch.full_like(node_embeddings, float('-inf')) # include_self=False leads to an error for some reason. Use -inf to get the same result.
        max_msg.index_reduce_(0, output_indices, output_messages, reduce='amax', include_self=True)
        return self._update_mlp(torch.cat((max_msg, node_embeddings), 1))


class SmoothMaximumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, problem: mm.Problem, embedding_size: int):
        super().__init__(problem, embedding_size)

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        exps_max = torch.zeros_like(node_embeddings)
        exps_max.index_reduce_(0, output_indices, output_messages, reduce="amax", include_self=False)
        exps_max = exps_max.detach()
        MAXIMUM_SMOOTHNESS = 12.0  # As the value approaches infinity, the hard maximum is attained
        max_offsets = exps_max.index_select(0, output_indices).detach()
        exps = (MAXIMUM_SMOOTHNESS * (output_messages - max_offsets)).exp()
        exps_sum = torch.full_like(node_embeddings, 1E-16)
        exps_sum.index_add_(0, output_indices, exps)
        max_msg = ((1.0 / MAXIMUM_SMOOTHNESS) * exps_sum.log()) + exps_max
        return self._update_mlp(torch.cat((max_msg, node_embeddings), 1))


class RelationalMessagePassingModule(nn.Module):
    def __init__(
        self,
        domain: mm.Domain,
        aggregation: str,
        embedding_size: int,
        num_layers: int,
        global_readout: bool = False,
        normalization: bool = False,
        random_initialization: bool = False,
    ):
        super().__init__()
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._global_readout = global_readout
        self._normalization = normalization
        self._random_initialization = random_initialization
        if aggregation == 'add':
            self._relation_network = SumRelationMessagePassing(domain, embedding_size)
        elif aggregation == 'mean':
            self._relation_network = MeanRelationMessagePassing(domain, embedding_size)
        elif aggregation == 'smax':
            self._relation_network = SmoothMaximumRelationMessagePassing(domain, embedding_size)
        elif aggregation == 'hmax':
            self._relation_network = HardMaximumRelationMessagePassing(domain, embedding_size)
        else:
            raise ValueError('aggregation is not one of ["add", "mean", "smax", "hmax"]')
        if global_readout:
            self._global_readout = SumReadout(embedding_size, embedding_size)
            self._global_update_mlp = MLP(2 * embedding_size, embedding_size)
        if normalization:
            self._normalization = nn.LayerNorm(embedding_size)

    def forward(
        self,
        relations: dict[str, torch.Tensor],
        object_indices: torch.Tensor,
        node_sizes: torch.Tensor,
        random_readout: bool,
    ) -> "tuple[torch.Tensor, Union[torch.Tensor, None]]":
        node_embeddings: torch.Tensor = torch.zeros([node_sizes.sum(), self._embedding_size], dtype=torch.float, requires_grad=True, device=node_sizes.device)
        if self._random_initialization:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)  # TODO: The seed should probably be the hash of the instance.
            random_embeddings = torch.randn([object_indices.size(0), self._embedding_size], dtype=torch.float, requires_grad=True, device=node_sizes.device)
            node_embeddings = node_embeddings.index_add(0, object_indices, random_embeddings)
            torch.set_rng_state(rng_state)
        random_iteration = random.randint(0, self._num_layers - 1) if random_readout else -1
        random_node_embeddings = None
        for iteration in range(self._num_layers):
            relation_messages = self._relation_network(node_embeddings, relations)
            if self._normalization:
                relation_messages = self._normalization(relation_messages)  # Normalize the magnitude of the message's values to be between -1 and 1.
            if self._global_readout:
                global_embedding = self._global_readout(node_embeddings, node_sizes)
                global_messages = self._global_update_mlp(torch.cat((node_embeddings, global_embedding.repeat_interleave(node_sizes, dim=0)), 1))
                if self._normalization:
                    global_messages = self._normalization(global_messages)
                node_embeddings = node_embeddings + global_messages + relation_messages
            else:
                node_embeddings = node_embeddings + relation_messages
            if random_iteration == iteration:
                random_node_embeddings = node_embeddings
        return node_embeddings, random_node_embeddings


class RelationalGraphNeuralNetwork(nn.Module):
    def __init__(self, domain: mm.Domain, aggregation: str, embedding_size: int, num_layers: int, global_readout: bool, normalization: bool, random_initialization: bool):
        """
        Relational Graph Neural Network (RGNN) for planning states.

        :param domain: The domain of the planning problem.
        :type domain: mimir.Domain
        :param aggregation: The aggregation method for message passing. Options are 'add', 'mean', 'smax', 'hmax'.
        :type aggregation: str
        :param embedding_size: The size of the node embeddings.
        :type embedding_size: int
        :param num_layers: The number of message passing layers.
        :type num_layers: int
        :param global_readout: Whether to use a global readout for the node embeddings.
        :type global_readout: bool
        :param normalization: Whether to apply layer normalization to the node embeddings.
        :type normalization: bool
        :param random_initialization: Whether to use random initialization for the node embeddings.
        :type random_initialization: bool
        """
        super().__init__()
        self._domain = domain
        self._aggregation = aggregation
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._global_readout = global_readout
        self._normalization = normalization
        self._random_initialization = random_initialization
        self._mpnn_module = RelationalMessagePassingModule(domain, aggregation, embedding_size, num_layers, global_readout, normalization, random_initialization)
        self._object_readout = SumReadout(embedding_size, embedding_size)
        self._action_value = MLP(2 * embedding_size, 1)
        self._value = MLP(embedding_size, 1)
        self._dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self._dummy.device

    def _create_value_input(self, states: list[mm.State]):
        # Get some stuff
        device = self.get_device()
        flattened_relations = {}
        object_indices = []
        object_sizes = []
        offset = 0
        # Helper function for populating relations and sizes.
        def add_atom_relation(atom: mm.GroundAtom, is_goal_atom: bool):
            relation_name = get_atom_name(atom, state, is_goal_atom)
            object_indices = [term.get_index() + offset for term in atom.get_terms()]
            if relation_name not in flattened_relations: flattened_relations[relation_name] = object_indices
            else: flattened_relations[relation_name].extend(object_indices)
        # Construct input
        for state in states:
            problem = state.get_problem()
            objects = problem.get_objects()
            num_objects = len(objects)
            # Add state relations
            for atom in state.get_ground_atoms():
                add_atom_relation(atom, False)
            # Add goal relations
            for ground_literal in problem.get_goal_condition():
                assert isinstance(ground_literal, mm.GroundLiteral), "Goal condition should contain ground literals."
                assert ground_literal.get_polarity(), "Only positive literals are supported."
                add_atom_relation(ground_literal.get_atom(), True)
            # Situation sizes
            object_indices.extend([object.get_index() + offset for object in objects])
            object_sizes.append(num_objects)
            offset += num_objects
        # Move input to device
        flattened_relations = relations_to_tensors(flattened_relations, device)
        object_indices = torch.tensor(object_indices, dtype=torch.int, device=device, requires_grad=False)
        object_sizes = torch.tensor(object_sizes, dtype=torch.int, device=device, requires_grad=False)
        return flattened_relations, object_indices, object_sizes

    def forward_value(self, states: list[mm.State], random_readout: bool = False) -> 'Union[list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]':
        """
        Computes the value for a list of states using the relational graph neural network.

        :param states: A list of states for which to compute the values.
        :type states: list[mm.State]
        :param random_readout: If True, returns the value for a randomly initialized node embedding as well.
        :type random_readout: bool
        :return: A list of values for each state. If random_readout is True, returns a tuple of two lists: the values for the final node embeddings and the values for a random layer's node embeddings.
        :rtype: Union[list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]
        """
        # Create input
        input, object_indices, object_sizes = self._create_value_input(states)
        # Readout function
        def value_readout(node_embeddings: torch.Tensor):
            object_embeddings = node_embeddings.index_select(0, object_indices)
            object_aggregation = self._object_readout(object_embeddings, object_sizes)
            return self._value(object_aggregation).view(-1)
        # Pass the input through the MPNN module
        final_node_embeddings, random_node_embeddings = self._mpnn_module.forward(input, object_indices, object_sizes, random_readout)
        if random_readout:
            return value_readout(final_node_embeddings), value_readout(random_node_embeddings)
        else:
            return value_readout(final_node_embeddings)

    def _create_q_values_input(self, states: list[mm.State]):
        # Get some stuff
        device = self.get_device()
        flattened_relations = {}
        node_sizes = []
        object_indices = []
        object_sizes = []
        action_indices = []
        action_sizes = []
        offset = 0
        # Helper function for populating relations and sizes.
        def add_atom_relation(atom: mm.GroundAtom, is_goal_atom: bool):
            relation_name = get_atom_name(atom, state, is_goal_atom)
            object_indices = [term.get_index() + offset for term in atom.get_terms()]
            if relation_name not in flattened_relations: flattened_relations[relation_name] = object_indices
            else: flattened_relations[relation_name].extend(object_indices)
        # Construct input
        for state in states:
            problem = state.get_problem()
            objects = problem.get_objects()
            actions = state.generate_applicable_actions()
            num_objects = len(objects)
            num_actions = len(actions)
            # Add state relations
            for atom in state.get_ground_atoms():
                add_atom_relation(atom, False)
            # Add goal relations
            for ground_literal in problem.get_goal_condition():
                assert isinstance(ground_literal, mm.GroundLiteral), "Goal condition should contain ground literals."
                assert ground_literal.get_polarity(), "Only positive literals are supported."
                add_atom_relation(ground_literal.get_atom(), True)
            # Add action relations
            for index, action in enumerate(actions):
                relation_name = get_action_name(action)
                action_id = num_objects + index
                action_indices.append(action_id + offset)
                term_ids = [action_id + offset] + [term.get_index() + offset for term in action.get_objects()]
                if relation_name not in flattened_relations: flattened_relations[relation_name] = term_ids
                else: flattened_relations[relation_name].extend(term_ids)
            # Situation sizes
            num_nodes = num_objects + num_actions
            node_sizes.append(num_nodes)
            # object_indices.extend(range(offset, offset + num_objects))  # Should be the the same thing.
            object_indices.extend([object.get_index() + offset for object in objects])
            object_sizes.append(num_objects)
            action_sizes.append(num_actions)
            offset += num_nodes
        # Move input to device
        flattened_relations = relations_to_tensors(flattened_relations, device)
        node_sizes = torch.tensor(node_sizes, dtype=torch.int, device=device, requires_grad=False)
        object_indices = torch.tensor(object_indices, dtype=torch.int, device=device, requires_grad=False)
        object_sizes = torch.tensor(object_sizes, dtype=torch.int, device=device, requires_grad=False)
        action_indices = torch.tensor(action_indices, dtype=torch.int, device=device, requires_grad=False)
        action_sizes = torch.tensor(action_sizes, dtype=torch.int, device=device, requires_grad=False)
        return flattened_relations, node_sizes, object_indices, object_sizes, action_indices, action_sizes

    def forward_q_values(self, states: list[mm.State], random_readout: bool = False) -> 'Union[list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]':
        """
        Computes the Q-values for a list of states using the relational graph neural network.

        :param states: A list of states for which to compute the Q-values.
        :type states: list[mm.State]
        :param random_readout: If True, returns the Q-values for a randomly initialized node embedding as well.
        :type random_readout: bool
        :return: A list of Q-values for each action in the states. If random_readout is True, returns a tuple of two lists: the Q-values for the final node embeddings and the Q-values for a random layer's node embeddings.
        :rtype: Union[list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]
        """
        # Create input
        input, node_sizes, object_indices, object_sizes, action_indices, action_sizes = self._create_q_values_input(states)
        # Readout function
        def q_value_readout(node_embeddings: torch.Tensor):
            action_embeddings = node_embeddings.index_select(0, action_indices)
            object_embeddings = node_embeddings.index_select(0, object_indices)
            object_aggregation = self._object_readout(object_embeddings, object_sizes)
            object_aggregation: torch.Tensor = object_aggregation.repeat_interleave(action_sizes, dim=0)
            values: torch.Tensor = self._action_value(torch.cat((action_embeddings, object_aggregation), dim=1))
            return [action_values.view(-1) for action_values in values.split(action_sizes.tolist())]
        # Pass the input through the MPNN module
        final_node_embeddings, random_node_embeddings = self._mpnn_module.forward(input, object_indices, node_sizes, random_readout)
        if random_readout:
            return q_value_readout(final_node_embeddings), q_value_readout(random_node_embeddings)
        else:
            return q_value_readout(final_node_embeddings)

    def forward(self) -> None:
        """
        This method is not implemented for RelationalGraphNeuralNetwork.
        Use forward_value or forward_q_values instead.
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("The forward method is not implemented for RelationalGraphNeuralNetwork. Use forward_value or forward_q_values instead.")

    def get_state_and_hparams_dicts(self) ->  'tuple[dict, dict]':
        """
        Returns the model's state dictionary and hyperparameters as a tuple.

        :return: A tuple containing the model's state dictionary and a dictionary of hyperparameters.
        :rtype: tuple[dict, dict]
        """
        hparams = {
            'aggregation': self._aggregation,
            'embedding_size': self._embedding_size,
            'num_layers': self._num_layers,
            'global_readout': self._global_readout,
            'normalization': self._normalization,
            'random_initialization': self._random_initialization
        }
        return self.state_dict(), hparams

    def clone(self) -> 'RelationalGraphNeuralNetwork':
        """
        Clones the model's weights and hyperparameters.

        :return: A new instance of RelationalGraphNeuralNetwork with the same weights and hyperparameters.
        :rtype: RelationalGraphNeuralNetwork
        """
        state_dict, hparams_dict = self.get_state_and_hparams_dicts()
        model_clone = RelationalGraphNeuralNetwork(self._domain, **hparams_dict)
        model_clone.load_state_dict(state_dict)
        return model_clone.to(self.get_device())

    def copy(self, destination_model: 'RelationalGraphNeuralNetwork') -> 'None':
        """
        Copies the model's weights to another instance of RelationalGraphNeuralNetwork.
        :param destination_model: The model to copy the weights to.
        :type destination_model: RelationalGraphNeuralNetwork
        """
        destination_model.load_state_dict(self.state_dict())

    def save(self, path: str, extras: dict = {}) -> 'None':
        """
        Saves the model's state and hyperparameters to a file.
        The parameter `extras` can be used to store additional information in the checkpoint, e.g., the optimizer state.

        :param path: The path to save the model checkpoint.
        :type path: str
        :param extras: Additional information to store in the checkpoint.
        :type extras: dict
        """
        model_dict, hparams_dict = self.get_state_and_hparams_dicts()
        checkpoint = { 'model': model_dict, 'hparams': hparams_dict, 'extras': extras}
        torch.save(checkpoint, path)

    @staticmethod
    def load(domain: mm.Domain, path: Union[Path, str], device: torch.device) -> 'tuple[RelationalGraphNeuralNetwork, dict]':
        """
        Loads a model from a checkpoint file.

        :param domain: The domain of the planning problem.
        :type domain: mimir.Domain
        :param path: The path to the model checkpoint file.
        :type path: str
        :param device: The device to load the model to (e.g., 'cpu' or 'cuda').
        :type device: torch.device
        :return: A tuple containing the loaded model and a dictionary with additional information (e.g., optimizer state).
        :rtype: tuple[RelationalGraphNeuralNetwork, dict]
        """
        checkpoint = torch.load(str(path), map_location=device)
        hparams_dict = checkpoint['hparams']
        model_dict = checkpoint['model']
        extras_dict = checkpoint['extras']
        model = RelationalGraphNeuralNetwork(domain, **hparams_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        return model, extras_dict
