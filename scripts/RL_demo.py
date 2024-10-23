import os

import ray
# from ray import tune
from gymnasium.envs.registration import register
from ray.rllib.algorithms.dqn import DQNConfig

# Import custom environment
from avp_env.envs.avp_env import AutonomousParkingEnv


# Register custom environment
register(
    id='AutonomousParking',
    entry_point='AVP_ENV:AutonomousParkingEnv',
)

# Initialise Ray
ray.init()

# Algorithm Configuration List
algorithm_configs = {
    "DQN": DQNConfig()
}

# Convolutional Filter Configuration
conv_filters_1 = [
    (32, 8, 4),
    (64, 4, 2),
    (64, 3, 1)
]
num_workers = 1

# Total time steps trained
total_timesteps_trained = 100000


def run_algorithm(algo_config, algo_name, total_timesteps):
    checkpoint_dir = f"../checkpoints/{algo_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    algo_config = algo_config.training(gamma=0.9, lr=0.01)
    algo_config = algo_config.resources(num_gpus=0)
    algo_config = algo_config.rollouts(num_rollout_workers=num_workers)
    algo_config = algo_config.environment(env=AutonomousParkingEnv)

    algo_config.replay_buffer_config["capacity"] = 10000  # reduce replay buffer

    # algo_config = algo_config.environment(env='AutonomousParking-v6')
    algo_config = algo_config.framework('torch')
    # algo_config = algo_config.model(conv_filters=conv_filters)
    algo_config.model["conv_filters"] = conv_filters_1

    algo = algo_config.build()

    timesteps = 0
    while timesteps < total_timesteps:
        result = algo.train()
        timesteps = result["timesteps_total"]
        rwd_mean = result['episode_reward_mean']
        len_mean = result['episode_len_mean']
        print("=*=" * 10)
        print(f"{algo_name} training at timestep {timesteps}/{total_timesteps}: {result}")
        print(f"|| Episode Reward Mean: {rwd_mean}, Episode Length Mean: {len_mean} ||")

        # Save checkpoints
        if timesteps % 10000 == 0:
            checkpoint = algo.save(checkpoint_dir)
            print(f"Checkpoint saved at: {checkpoint}")


# Configure and run Benchmark for each online algorithm
for algorithm_name, algorithm_config in algorithm_configs.items():
    run_algorithm(
        algorithm_config,
        algorithm_name,
        total_timesteps_trained
    )

ray.shutdown()
