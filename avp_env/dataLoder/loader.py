import json
import os

from avp_env.common import Trajectory, Instruction, ParkingSlot
from avp_env.dataLoder.path import PathLoader


class DataReader:
    def __init__(self, env_type):
        self.path_loader = PathLoader(env_type)
        self.experiment_paths = self.path_loader.load_path()

    @staticmethod
    def _load_json(json_path, filename):
        json_path = os.path.join(json_path, filename)
        with open(json_path, 'r') as f:
            return json.load(f)

    def load_parking_slots(self):
        combined_parking_data = []

        for experiment_path in self.experiment_paths:
            parking_data = self._load_json(experiment_path, 'parking_slots.json')
            combined_parking_data.extend(parking_data)

            # parking_data_sorted = sorted(parking_data, key=lambda x: x["ParkingID"])
        combined_parking_data_sorted = sorted(combined_parking_data, key=lambda x: x["ParkingID"])

        return [ParkingSlot(slot_data) for slot_data in combined_parking_data_sorted]

    def load_trajectories(self):
        combined_traj_data = []
        for experiment_path in self.experiment_paths:
            traj_data = self._load_json(experiment_path, 'Traj.json')
            combined_traj_data.extend(traj_data)

        return [Trajectory(traj_entry) for traj_entry in combined_traj_data]

    def load_metrics_instructions(self, env_type):
        if env_type == 'train':
            instruction_name = 'target_command.json'
        elif env_type == 'test':
            instruction_name = 'test_command.json'
        else:
            instruction_name = ''
        instruction_data = self._load_json('../data/commands', instruction_name)

        return [Instruction(instruction_entry) for instruction_entry in instruction_data]
