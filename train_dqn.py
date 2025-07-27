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
            goal = state.get_problem().get_goal_condition()
            input_list.append((state, actions, goal))
            actions_list.append(actions)
        q_values_list: list[torch.Tensor] = self.model.forward(input_list).readout('q')  # type: ignore
        output = list(zip(q_values_list, actions_list))
        for tensor, _ in output:
            assert not tensor.isnan().any()
            assert not tensor.isinf().any()
        return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument('--input', required=True, type=Path, help='Path to the training dataset')
    parser.add_argument('--model', default=None, type=Path, help='Path to a pre-trained model to continue training from')
    parser.add_argument('--embedding_size', default=32, type=int, help='Dimension of the embedding vector for each object')
    parser.add_argument('--aggregation', default='smax', type=str, help='Aggregation function ("smax", "hmax", "sum", or "mean")')
    parser.add_argument('--layers', default=30, type=int, help='Number of layers in the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per batch')
    parser.add_argument('--train_steps', default=16, type=int, help='Number of train steps per episode')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate for the training process')
    parser.add_argument('--discount_factor', default=0.999, type=float, help='Discount factor for the loss function')
    parser.add_argument('--horizon', default=100, type=int, help='Maximum trajectory length during rollout')
    parser.add_argument('--bt_initial', default=1.0, type=float, help='Initial Boltzmann temperature')
    parser.add_argument('--bt_final', default=0.1, type=float, help='Final Boltzmann temperature')
    parser.add_argument('--bt_steps', default=600, type=int, help='Number of steps for the Boltzmann temperature to decrease from the initial value to the final value')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args


def _parse_instances(input: Path) -> tuple[mm.Domain, list[mm.Problem]]:
    print('Finding instances...', flush=True)
    if input.is_file():
        domain_path = str(input.parent / 'domain.pddl')
        problem_paths = [str(input)]
    else:
        domain_path = str(input / 'domain.pddl')
        problem_paths = [str(file) for file in input.glob('*.pddl') if file.name != 'domain.pddl']
        problem_paths.sort()
    print('Parsing instances...', flush=True)
    domain = mm.Domain(domain_path)
    problems = [mm.Problem(domain, problem_path) for problem_path in problem_paths]
    return domain, problems


def _create_model(domain: mm.Domain, embedding_size: int, num_layers: int, aggregation: str, device: torch.device) -> rgnn.RelationalGraphNeuralNetwork:
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
    return rgnn.RelationalGraphNeuralNetwork(config).to(device)


def _train(model: rgnn.RelationalGraphNeuralNetwork,
           optimizer: torch.optim.Optimizer,
           lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
           problems: list[mm.Problem],
           args: argparse.Namespace):
    bt_initial = args.bt_initial
    bt_final = args.bt_final
    bt_steps = args.bt_steps
    horizon = args.horizon
    train_steps = args.train_steps
    discount_factor = args.discount_factor
    wrapped_model = ModelWrapper(model)
    loss_function = rl.DQNLossFunction(discount_factor, 100.0)
    reward_function = rl.ConstantRewardFunction(-1)
    replay_buffer = rl.PrioritizedReplayBuffer(1000)
    trajectory_sampler = rl.BoltzmannTrajectorySampler(1.0)
    problem_sampler = rl.UniformProblemSampler()
    initial_state_sampler = rl.OriginalInitialStateSampler()
    goal_sampler = rl.OriginalGoalConditionSampler()
    trajectory_refiner = rl.LiftedHindsightTrajectoryRefiner(problems, 100)
    rl_algorithm = rl.OffPolicyAlgorithm(problems,
                                         wrapped_model,
                                         optimizer,
                                         lr_scheduler,
                                         loss_function,
                                         reward_function,
                                         replay_buffer,
                                         trajectory_sampler,
                                         horizon,
                                         train_steps,
                                         problem_sampler,
                                         initial_state_sampler,
                                         goal_sampler,
                                         trajectory_refiner)
    evaluation_criteras = [rl.CoverageCriteria(), rl.SolutionLengthCriteria()]
    evaluation_trajectory_sampler = rl.GreedyPolicyTrajectorySampler()
    rl_evaluator = rl.PolicyEvaluation(problems, evaluation_criteras, evaluation_trajectory_sampler, reward_function, 4 * horizon)
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
    rl_algorithm.register_train_step(lambda ts, l1, l2, l3:
                                     print(f'[{episode}] > Train step: {l1.mean().item():.5f} avg. prediction; {l2.mean().item():.5f} avg. TD-error; {l3.mean().item():.5f} avg. bounds error.'))
    rl_algorithm.register_post_optimize_model(lambda: print(f'[{episode}] Optimized Model.', flush=True))
    while True:
        # Update Boltzmann temperature.
        bt_ratio = min(1.0, episode / bt_steps)
        bt_temp = bt_ratio * bt_final + (1.0 - bt_ratio) * bt_initial
        trajectory_sampler.set_temperature(bt_temp)
        print(f'[{episode}] Boltzmann Exploration: {bt_temp:.5f}', flush=True)
        # Run RL algorithm.
        rl_algorithm.fit(4, args.batch_size)
        # Evaluate every now and then.
        best, evaluation = rl_evaluator.evaluate(wrapped_model)
        print(f'[{episode}] Best: {best}, Evaluation: {evaluation}', flush=True)
        # Increment episode.
        episode += 1


def _main(args: argparse.Namespace) -> None:
    print(f'Torch: {torch.__version__}', flush=True)
    device = create_device()
    domain, problems = _parse_instances(args.input)
    print('Creating model...', flush=True)
    model = _create_model(domain, args.embedding_size, args.layers, args.aggregation, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 600)
    print('Training model...', flush=True)
    _train(model, optimizer, lr_scheduler, problems, args)


if __name__ == "__main__":
    _main(_parse_arguments())
