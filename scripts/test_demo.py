import json
# import numpy as np
import os
import zipfile

from avp_env.agents.rule import RulebasedAgent
from avp_env.envs.avp_env import MetricsEnv


def get_result_id(env, agent, instructions_index=None):
    env.reset(instructions_index)

    done = False
    steps = 0
    perfect_trajectory = env.get_perfect_traj()

    while not done:
        # TODO: change to your agent
        current_position = env.get_position()
        action = agent.get_action(perfect_trajectory, current_position)  # 获取动作
        observation, reward, done, info = env.step(action)  # 执行动作
        steps += 1

    last_slots = env.get_current_parking_slot()

    if last_slots:
        last_slot = last_slots[0]
        result_id = last_slot.ParkingID
    else:
        result_id = []

    return result_id


def get_experiment(env, agent, instru_num):
    experiments = []
    for instructions_index in range(instru_num):
        result_id = get_result_id(
            env=env, agent=agent,
            instructions_index=instructions_index
        )
        # target_id = get_target_id(env)
        experiment = {
            "TestScenarioID": env.get_scan(),
            "TestInstructionID": instructions_index,
            "VLPDecisionPositionID": result_id
        }
        experiments.append(experiment)
    return experiments


def instru_len(instru_path):
    with open(instru_path, 'r') as f:
        instru_data = json.load(f)
    return len(instru_data)


if __name__ == "__main__":
    # 创建 AutonomousParkingEnv 环境实例
    env = MetricsEnv()

    is_optimal = False
    is_random = True
    agent = RulebasedAgent(is_optimal, is_random)

    instruction_path = '../data/commands/test_command.json'
    instruction_num = instru_len(instruction_path)

    experiments = get_experiment(env, agent, instruction_num)

    # save experiments as JSON
    json_filename = '../result/test_results.json'
    json_folder = os.path.dirname(json_filename)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    with open(json_filename, 'w') as json_file:
        json.dump(experiments, json_file, indent=4)

    # save json as zip
    zip_filename = '../result/test_results.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_filename)

    print(f"save {json_filename} and zip as {zip_filename}")
