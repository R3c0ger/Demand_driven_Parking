import random

from gymnasium import spaces


class RandomAgent:
    def __init__(self):
        self.action_space = spaces.Discrete(7)  # discrete action space

    def get_action(self, observation):
        return self.action_space.sample() if random.random() <= 0.1 else 0


class RulebasedAgent:
    def __init__(self, is_optimal=False, is_random=False):
        if is_optimal and is_random:
            self.mode = "Good"     # 10% Random, 90% Optimal
        elif is_optimal:
            self.mode = "Optimal"  # 0% Random, 100% Optimal
        elif is_random:
            self.mode = "Random"   # 100% Random, 0% Optimal
        else:
            self.mode = "Normal"   # 50% Random, 50% Optimal
        self.action_space = spaces.Discrete(7)  # discrete action space

    def get_action(self, perfect_trajectory, current_position):
        random_action = self.action_space.sample()

        if self.mode == "Good":
            try:
                # Try to get optimal_action, throw an exception if current_position is invalid
                optimal_action = perfect_trajectory[current_position - 1]
            except IndexError:
                raise ValueError(
                    f"Invalid current_position: {current_position}. "
                    f"It must be between 1 and {len(perfect_trajectory)}"
                )
            action = random_action if random.random() <= 0.1 else optimal_action
        elif self.mode == "Optimal":
            try:
                # Try to get optimal_action, throw an exception if current_position is invalid
                # print(current_position)
                optimal_action = perfect_trajectory[current_position - 1]
            except IndexError:
                raise ValueError(
                    f"Invalid current_position: {current_position}. "
                    f"It must be between 1 and {len(perfect_trajectory)}"
                )
            action = optimal_action
        elif self.mode == "Random":
            action = random_action
        else:
            try:
                # Try to get optimal_action, throw an exception if current_position is invalid
                optimal_action = perfect_trajectory[current_position - 1]
            except IndexError:
                raise ValueError(
                    f"Invalid current_position: {current_position}. "
                    f"It must be between 1 and {len(perfect_trajectory)}"
                )

            action = random_action if random.random() <= 0.5 else optimal_action
        return action
