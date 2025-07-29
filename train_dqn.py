import argparse
import pymimir as mm
import pymimir_rgnn as rgnn
import pymimir_rl as rl
import torch
import torch.optim as optim

from pathlib import Path
from utils import create_device


class ModelWrapper(rl.QValueModel):
    def __init__(self, model: rgnn.RelationalGraphNeuralNetwork) -> None:
        super().__init__()
        self.model = model

    def forward(self, state_goals: list[tuple[mm.State, mm.GroundConjunctiveCondition]]) -> list[tuple[torch.Tensor, list[mm.GroundAction]]]:
        input_list: list[tuple[mm.State, list[mm.GroundAction], mm.GroundConjunctiveCondition]] = []
        actions_list: list[list[mm.GroundAction]] = []
        for state, goal in state_goals:
            actions = state.generate_applicable_actions()
            input_list.append((state, actions, goal))
            actions_list.append(actions)
        q_values_list: list[torch.Tensor] = self.model.forward(input_list).readout('q')  # type: ignore
        output = list(zip(q_values_list, actions_list))
        for tensor, _ in output:
            assert not tensor.isnan().any()
            assert not tensor.isinf().any()
        return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for training with DQN')
    parser.add_argument('--train', required=True, type=Path, help='Path to directory with training instances')
    parser.add_argument('--validation', required=True, type=Path, help='Path to directory with validation instances')
    parser.add_argument('--hindsight', required=True, type=str, choices=['lifted', 'propositional', 'state', 'state_fluent'], help='Type of hindsight to use')
    parser.add_argument('--model', default=None, type=Path, help='Path to the model file to resume from')
    parser.add_argument('--aggregation', default='hmax', type=str, help='Aggregation function used by the model ("add", "mean", "smax", "hmax")')
    parser.add_argument('--embedding_size', default=32, type=int, help='Dimension of the embedding vector for each object')
    parser.add_argument('--layers', default=60, type=int, help='Number of layers in the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per batch')
    parser.add_argument('--bt_initial', default=1.0, type=float, help='Initial Boltzmann temperature')
    parser.add_argument('--bt_final', default=0.1, type=float, help='Final Boltzmann temperature')
    parser.add_argument('--bt_steps', default=600, type=int, help='Number of steps for the Boltzmann temperature to decrease from the initial value to the final value')
    parser.add_argument('--discount_factor', default=0.999, type=float, help='Discount factor')
    parser.add_argument('--train_horizon', default=100, type=int, help='Maximum rollout length for the training set')
    parser.add_argument('--validation_horizon', default=400, type=int, help='Maximum rollout length for the validation set')
    parser.add_argument('--lr_initial', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--lr_final', default=0.000001, type=float, help='Final learning rate')
    parser.add_argument('--lr_steps', default=300, type=float, help='Steps to reach the final learning rate')
    parser.add_argument('--max_new_trajectories', default=100, type=int, help='Max number of new trajectories to derive')
    parser.add_argument('--min_buffer_size', default=100, type=int, help='Minimum size of the experience buffer to update model')
    parser.add_argument('--max_buffer_size', default=1000, type=int, help='Maximum size of the experience buffer')
    parser.add_argument('--num_rollouts', default=4, type=int, help='Number of trajectories to compute in parallel')
    parser.add_argument('--train_steps', default=32, type=int, help='Number of training steps per iteration')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--cpu', action='store_true', help='Force CPU to be used')
    args = parser.parse_args()
    return args


def _parse_instances(input: Path) -> tuple[mm.Domain, list[mm.Problem]]:
    if input.is_file():
        domain_path = str(input.parent / 'domain.pddl')
        problem_paths = [str(input)]
    else:
        domain_path = str(input / 'domain.pddl')
        problem_paths = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_paths.sort()
    domain = mm.Domain(domain_path)
    problems = [mm.Problem(domain, problem_path) for problem_path in problem_paths]
    return domain, problems


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int, aggregation: str) -> rgnn.RelationalGraphNeuralNetwork:
    if aggregation == 'smax': aggregation_type = rgnn.AggregationFunction.SmoothMaximum
    elif aggregation == 'hmax': aggregation_type = rgnn.AggregationFunction.HardMaximum
    elif aggregation == 'mean': aggregation_type = rgnn.AggregationFunction.Mean
    elif aggregation == 'add': aggregation_type = rgnn.AggregationFunction.Add
    else: raise RuntimeError(f'Unknown aggregation function: {aggregation}.')
    config = rgnn.RelationalGraphNeuralNetworkConfig(
        domain=domain,
        input_specification=(rgnn.InputType.State, rgnn.InputType.GroundActions, rgnn.InputType.Goal),
        output_specification=[('q', rgnn.OutputNodeType.Action, rgnn.OutputValueType.Scalar)],
        embedding_size=embedding_size,
        num_layers=num_layers,
        message_aggregation=aggregation_type
    )
    return rgnn.RelationalGraphNeuralNetwork(config)


def _train(model: rgnn.RelationalGraphNeuralNetwork,
           optimizer: torch.optim.Optimizer,
           lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
           train_problems: list[mm.Problem],
           validation_problems: list[mm.Problem],
           args: argparse.Namespace,
           device: torch.device):
    wrapped_model = ModelWrapper(model).to(device)
    loss_function = rl.DQNLossFunction(wrapped_model, args.discount_factor, 10.0)
    reward_function = rl.ConstantRewardFunction(-1)
    replay_buffer = rl.PrioritizedReplayBuffer(args.max_buffer_size)
    trajectory_sampler = rl.BoltzmannTrajectorySampler(1.0)
    problem_sampler = rl.UniformProblemSampler()
    initial_state_sampler = rl.OriginalInitialStateSampler()
    goal_sampler = rl.OriginalGoalConditionSampler()
    trajectory_refiner = rl.LiftedHindsightTrajectoryRefiner(train_problems, args.max_new_trajectories)
    rl_algorithm = rl.OffPolicyAlgorithm(train_problems,
                                         wrapped_model,
                                         optimizer,
                                         lr_scheduler,
                                         loss_function,
                                         reward_function,
                                         replay_buffer,
                                         trajectory_sampler,
                                         args.train_horizon,
                                         args.num_rollouts,
                                         args.batch_size,
                                         args.train_steps,
                                         problem_sampler,
                                         initial_state_sampler,
                                         goal_sampler,
                                         trajectory_refiner)
    evaluation_criteras = [rl.CoverageCriteria(), rl.SolutionLengthCriteria()]
    evaluation_trajectory_sampler = rl.GreedyPolicyTrajectorySampler()
    rl_evaluator = rl.PolicyEvaluation(validation_problems, evaluation_criteras, evaluation_trajectory_sampler, reward_function, args.validation_horizon)
    episode = 0
    def avg_num_objects(ps: list[mm.Problem]) -> float:
        return sum(len(p.get_objects()) for p in ps) / len(ps)
    def avg_goal_size(ts: list[rl.Trajectory]) -> float:
        return sum(len(t[0].goal_condition) for t in ts if len(t) > 0) / len(ts)
    def avg_trajectory_length(ts: list[rl.Trajectory]) -> float:
        return sum(len(t) for t in ts if len(t) > 0) / len(ts)
    rl_algorithm.register_pre_collect_experience(lambda: print(f'[{episode}] Collecting Experience.', flush=True))
    rl_algorithm.register_sample_problems(lambda ps: print(f'[{episode}] > Sampled Problems; {avg_num_objects(ps):.1f} avg. object count.', flush=True))
    rl_algorithm.register_sample_initial_states(lambda x: print(f'[{episode}] > Sampled Initial States.', flush=True))
    rl_algorithm.register_sample_goal_conditions(lambda x: print(f'[{episode}] > Sampled Goals.', flush=True))
    rl_algorithm.register_sample_trajectories(lambda ts: print(f'[{episode}] > Sampled Trajectories; {avg_goal_size(ts):.1f} avg. goal size; {avg_trajectory_length(ts):.1f} avg. trajectory length', flush=True))
    rl_algorithm.register_refine_trajectories(lambda ts: print(f'[{episode}] > Refined Trajectories; {avg_goal_size(ts):.1f} avg. goal size; {avg_trajectory_length(ts):.1f} avg. trajectory length.', flush=True))
    rl_algorithm.register_post_collect_experience(lambda: print(f'[{episode}] Collected Experience.', flush=True))
    rl_algorithm.register_pre_optimize_model(lambda: print(f'[{episode}] Optimizing Model.', flush=True))
    rl_algorithm.register_train_step(lambda ts, l1, l2, l3: print(f'[{episode}] > Train step: {l1.mean().item():.5f} avg. prediction; {l2.mean().item():.5f} avg. TD-error; {l3.mean().item():.5f} avg. bounds error.'))
    rl_algorithm.register_post_optimize_model(lambda: print(f'[{episode}] Optimized Model.', flush=True))
    while True:
        # Update Boltzmann temperature.
        bt_ratio = min(1.0, episode / args.bt_steps)
        bt_temp = bt_ratio * args.bt_final + (1.0 - bt_ratio) * args.bt_initial
        trajectory_sampler.set_temperature(bt_temp)
        print(f'[{episode}] Boltzmann Exploration: {bt_temp:.5f}', flush=True)
        # Run RL algorithm.
        rl_algorithm.fit()
        # Evaluate every now and then.
        best, evaluation = rl_evaluator.evaluate(wrapped_model)
        print(f'[{episode}] Best: {best}, Evaluation: {evaluation}', flush=True)
        # Increment episode.
        episode += 1


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}', flush=True)
    device = create_device(args.cpu)
    domain, train_problems = _parse_instances(args.train)
    print(f'Parsed {len(train_problems)} training instances.', flush=True)
    _, validation_problems = _parse_instances(args.validation)
    print(f'Parsed {len(validation_problems)} validation instances.', flush=True)
    print('Creating model...', flush=True)
    model = _create_model(domain, args.embedding_size, args.layers, args.aggregation)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_initial)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_steps, args.lr_final)
    print('Training model...', flush=True)
    _train(model, optimizer, lr_scheduler, train_problems, validation_problems, args, device)


if __name__ == "__main__":
    _main(_parse_arguments())
