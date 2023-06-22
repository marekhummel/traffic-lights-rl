from typing import Any
import gymnasium as gym
import numpy as np

from util import ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON, Action, Observation


class TrafficLightEnv(gym.Env):
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode: str = "console") -> None:
        super(TrafficLightEnv, self).__init__()
        self.render_mode = render_mode

        # Params
        self.num_lights = 2
        self.cars_passing_per_step = 4
        self.car_arrival_dist = [0] * 6 + [1] * 9 + [2] * 6 + [3] * 3 + [4] * 2 + [5] * 0

        # State
        _ = self.reset()

        # Spaces
        self.action_space = gym.spaces.Discrete(self.num_lights + 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed: Any = None, options: Any = None) -> tuple[Observation, dict]:
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        self.cars_left: list[int] = [0]
        self.cars_right: list[int] = [0]
        return self._to_state(), {}  # empty info dict

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        if action not in {ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON}:
            raise ValueError(f"Received invalid action={action} which is not part of the action space")

        # Compute reward
        reward = 0.0
        info = {"passed_left": 0, "passed_right": 0}

        # Let cars pass
        if action == LEFT_LIGHT_ON:
            passing = min(len(self.cars_left), self.cars_passing_per_step)
            self.cars_left = self.cars_left[passing:]
            reward += passing
            info["passed_left"] = passing
        elif action == RIGHT_LIGHT_ON:
            passing = min(len(self.cars_right), self.cars_passing_per_step)
            self.cars_right = self.cars_right[passing:]
            reward += passing
            info["passed_right"] = passing

        # Waiting cars reward
        all_waiting_cars = self.cars_left + self.cars_right
        if all_waiting_cars:
            # reward -= len(all_waiting_cars)
            # reward -= max(all_waiting_cars) ** 2
            reward -= sum(all_waiting_cars)
            # reward -= sum(all_waiting_cars) / len(all_waiting_cars)

        # Waiting cars wait time
        self.cars_left = [c + 1 for c in self.cars_left]
        self.cars_right = [c + 1 for c in self.cars_right]

        # New cars coming
        new_left, new_right = np.random.choice(self.car_arrival_dist), np.random.choice(self.car_arrival_dist)
        info.update(new_left=new_left, new_right=new_right)
        self.cars_left.extend([0] * new_left)
        self.cars_right.extend([0] * new_right)

        # Finished ?
        terminated = bool(len(self.cars_left) == 0 and len(self.cars_right) == 0)
        truncated = False

        # Return
        return (
            self._to_state(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            left = len(self.cars_left)  # " ".join([str(c) for c in reversed(self.cars_left)])
            right = len(self.cars_right)  # " ".join([str(c) for c in self.cars_right])
            print(f"{left} -> |    | <- {right}")

    def close(self) -> None:
        pass

    def _to_state(self) -> Observation:
        def light_state(cars: list[int]) -> tuple[float, float, float]:
            # return (len(cars), max(cars), sum(cars), sum(cars) / len(cars)) if cars else (0, 0, 0, 0)
            return (sum(cars), len(cars)) if cars else (0, 0)

        return np.array(light_state(self.cars_left) + light_state(self.cars_right)).astype(np.float32)


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = TrafficLightEnv()
    check_env(env, warn=True)
    print("Env checked successfully")
