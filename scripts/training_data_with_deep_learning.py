import h5py
import numpy as np
import pandas as pd
from avp_env.envs import AutonomousParkingEnv
from avp_env.agents.rule import RulebasedAgent


def collect_data_for_supervised_learning(env, agent, num_episodes=100):
    """
    Collecting data using Agents in the environment, suitable for supervised learning
    """
    images = []
    instructions = []
    actions = []
    for _ in range(num_episodes):
        obs = env.reset()
        perfect_trajectory = env.getPerfectTraj()
        for i in range(len(perfect_trajectory)):
            image, instruction = obs
            current_position = env.getPosition()
            action = agent.get_action(perfect_trajectory, current_position)

            next_obs, _, done, info = env.step(action)
            if i != 0:
                images.append(image)  # Flatten the image for easier storage
                instructions.append(instruction)
                actions.append(action)

            obs = next_obs
            if done:
                break
    return np.array(images), np.array(instructions), np.array(actions)


# Creating an Instance of the AutonomousParkingEnv Environment
env = AutonomousParkingEnv()
isOptimal = True
isRandom = False

agent = RulebasedAgent(isOptimal=isOptimal, isRandom=isRandom)

# Collection of data
images, instructions, actions = collect_data_for_supervised_learning(env, agent, num_episodes=1000)

# Save the dataset as HDF5 files
with h5py.File(f'New_Supervised_Opt_{isOptimal}_Ran_{isRandom}_dataset.h5', 'w') as f:
    f.create_dataset('images', data=images)
    f.create_dataset('instructions', data=instructions)
    f.create_dataset('actions', data=actions)
