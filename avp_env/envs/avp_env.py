import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer
from avp_env.dataLoder import ImageLoader, DataReader


class AutonomousParkingEnv(gym.Env):
    def __init__(self, args=[]):
        super(AutonomousParkingEnv, self).__init__()
        self.env_type = 'train'
        self.image_shape = (128, 400, 3)
        self.max_string_length = 64

        # Initialize helpers
        self.image_loader = ImageLoader(self.env_type, self.image_shape)
        self.data_reader = DataReader(self.env_type)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Initialize environment data
        self.image_data = self.image_loader.image_data
        self.render_image = self.image_loader.render_image
        self.parking_slots = self.data_reader.load_parking_slots()
        self.trajectories = self.data_reader.load_trajectories()
        self.metrics_instructions = self.data_reader.load_metrics_instructions(self.env_type)

        # Define observation space
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
            spaces.Box(low=0, high=99999, shape=(self.max_string_length,), dtype=np.int64)
        ))

        # Define action space
        self.action_space = spaces.Discrete(7)

        # Initialize state
        self.current_observation = (
            np.zeros(self.image_shape, dtype=np.uint8), np.zeros(self.max_string_length, dtype=np.uint8))

    def get_parking_slots(self, loc_id, path_id):
        return [slot for slot in self.parking_slots if slot.LocID == loc_id and slot.PathID == path_id]

    def get_perfect_trajectory(self, trajectory):
        if hasattr(trajectory, "path_id"):
            perfect_trajectory = [0] * (trajectory.path_id - 1)
            perfect_trajectory.append(trajectory.loc_id)
        else:
            perfect_trajectory = [0] * 29
        return perfect_trajectory

    # def reset(self, InsIndex=None):
    #     self.current_position = 1
    #     self.target_instruction = random.choice(self.trajectories)
    #     instruction_tokens = self.tokenizer.encode(
    #         self.target_instruction.instruction, add_special_tokens=True,
    #         max_length=self.max_string_length, pad_to_max_length=True,
    #         truncation=True
    #     )
    #
    #     self.inital_instruction = np.array(instruction_tokens)
    #
    #     # self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction.loc_id, self.target_instruction.path_id)
    #
    #     self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction)
    #
    #     self.render_observation = self.render_image[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"]
    #     self.current_observation = (
    #         self.image_data[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"], self.inital_instruction)
    #     return self.current_observation

    def reset(self, InsIndex=None):
        self.current_position = 1
        # self.target_instruction = random.choice(self.trajectories)
        self.target_instruction = self.trajectories[9]
        instruction_tokens = self.tokenizer.encode(
            self.target_instruction.instruction, add_special_tokens=True,
            max_length=self.max_string_length, pad_to_max_length=True,
            truncation=True
        )

        self.inital_instruction = np.array(instruction_tokens)

        # self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction.loc_id, self.target_instruction.path_id)

        self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction)

        self.render_observation = self.render_image[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"]
        self.current_observation = (
            self.image_data[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"], self.inital_instruction)
        return self.current_observation

    def getPerfectTraj(self):
        return self.perfect_trajectory

    def getPosition(self):
        return self.current_position

    def getCurrentParkingSlot(self):
        return self.CurrentParkingSlot

    def getTargetInstruction(self):
        return self.target_instruction

    def step(self, action):

        # 执行动作并返回奖励、下一个观察、是否终止、调试信息
        if self.current_position > 29:
            reward = -1
            done = True
            self.CurrentParkingSlot = []
        elif action == 0 and self.current_position != 29:
            reward = 0
            self.current_position += 1
            done = False
        elif action == 0 and self.current_position == 29:
            reward = -1
            done = True
            self.CurrentParkingSlot = []
        else:
            self.CurrentParkingSlot = self.get_parking_slots(action, self.current_position)
            done = True
            if hasattr(self.target_instruction, 'ParkingID'):
                reward = self.getReward(self.CurrentParkingSlot)
            else:
                reward = 0

        # # 执行动作并返回奖励、下一个观察、是否终止、调试信息
        # if self.current_position > 29:
        #     reward = -1
        #     done = True
        #     self.CurrentParkingSlot = []
        # elif action == 0 and self.current_position != 29:
        #     reward = 0
        #     self.current_position += 1
        #     done = False
        # elif action != 0:
        #     self.CurrentParkingSlot = self.get_parking_slots(action, self.current_position)
        #     done = True
        #     if hasattr(self.target_instruction, 'ParkingID'):
        #         reward = self.getReward(self.CurrentParkingSlot)
        #     else:
        #         reward = 0
        # else:
        #     reward = -1
        #     done = True
        #     self.CurrentParkingSlot = []

        self.render_observation = self.render_image[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"]

        self.current_observation = (self.image_data[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"],
                                    self.inital_instruction)
        # self.current_observation = (np.zeros(self.image_shape, dtype=np.uint8), np.zeros(self.max_string_length, dtype=np.uint8))

        info = {}  # 可以用来传递额外的调试信息

        return self.current_observation, reward, done, info

    def getReward(self, CurrentParkingSlot):
        if CurrentParkingSlot:
            for slot in CurrentParkingSlot:
                if slot.ParkingID == self.target_instruction.ParkingID:
                    reward = 10  # 如果当前停车位与目标停车位相同，给一个很大的奖励
                elif slot.Occupied != 0:
                    reward = -0.5  # 不空的车位，给一个负的惩罚性奖励:
                elif slot.Disabled != self.target_instruction.tags['Disabled']:
                    reward = -0.2  # 停错残疾人车位，给一个负的惩罚性奖励:

                elif slot.Charging != self.target_instruction.tags['Charging']:
                    reward = -0.2  # 停错充电车位，给一个负的惩罚性奖励:

                else:
                    reward = 5
                    for key, value in self.target_instruction.tags.items():
                        if getattr(slot, key, None) == value:
                            reward += 0.2  # 如果 slot 中的属性与目标属性相同，给一个中等的奖励
        else:
            reward = -1  # 决策了一个不存在的车位，给一个负的惩罚性奖励

        return reward

    def render(self, mode='human'):
        self.render_observation[:, :, [0, 2]] = self.render_observation[:, :, [2, 0]]
        # 可选的渲染方法，用于可视化环境状态
        img, command = self.render_observation, self.target_instruction.instruction
        return img, command

    def close(self):
        # 关闭环境，释放资源
        pass


class MetricsEnv(AutonomousParkingEnv):
    def __init__(self, env_type='train'):
        super(MetricsEnv, self).__init__(env_type)
        self.trajectory_index = 0  # Initialize trajectory index
        self.traj_len = len(self.trajectories)

    def reset(self, InsIndex=None):

        self.current_position = 1
        if InsIndex == None:
            # Select the next trajectory in sequence
            self.target_instruction = self.trajectories[self.trajectory_index]
            self.trajectory_index = (self.trajectory_index + 1) % len(self.trajectories)
        else:
            self.target_instruction = self.trajectories[InsIndex]

        instruction_tokens = self.tokenizer.encode(
            self.target_instruction.instruction, add_special_tokens=True,
            max_length=self.max_string_length, pad_to_max_length=True,
            truncation=True
        )

        self.inital_instruction = np.array(instruction_tokens)
        self.perfect_trajectory = self.get_perfect_trajectory(self.target_instruction)
        self.render_observation = self.render_image[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"]
        self.current_observation = (
            self.image_data[f"{self.target_instruction.scan}/DJI_{self.current_position}.JPG"], self.inital_instruction)

        return self.current_observation
