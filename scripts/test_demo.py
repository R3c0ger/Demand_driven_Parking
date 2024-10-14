import numpy as np
from avp_env.envs.avp_env import MetricsEnv
from avp_env.agents.rule import RulebasedAgent
import json
import zipfile
import os

def getResultID(env, agent, instructions_index=None):

    env.reset(instructions_index)

    done = False
    steps = 0
    perfect_trajectory = env.getPerfectTraj()

    while not done:
        # todo change to your agent
        current_position = env.getPosition()
        action = agent.get_action(perfect_trajectory, current_position)  # 获取动作
        observation, reward, done, info = env.step(action)  # 执行动作
        steps += 1

    last_slots = env.getCurrentParkingSlot()


    if last_slots:
        last_slot = last_slots[0]
        result_id = last_slot.ParkingID
    else:
        result_id = []

    return result_id

def get_experiment(env, agent, instru_num):
    experiments = []
    for instructions_index in range(instru_num):
        result_id = getResultID(env=env, agent=agent, instructions_index=instructions_index)
        # target_id = getTargetID(env)

        experiment = {
            "TestScenarioID": env.getScan(),
            "TestInstructionID": instructions_index,
            "VLPDecisionPositionID": result_id
        }

        experiments.append(experiment)

    return experiments

def instru_len(instruction_path):
    with open(instruction_path, 'r') as f:
        instruction_data = json.load(f)
    return len(instruction_data)

if __name__ == "__main__":
    # 创建 AutonomousParkingEnv 环境实例
    env = MetricsEnv()
    isOptimal = False

    isRandom = True
    agent = RulebasedAgent(isOptimal, isRandom)

    instruction_path = '../data/commands/test_command.json'
    instru_num = instru_len(instruction_path)

    experiments = get_experiment(env, agent, instru_num)
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