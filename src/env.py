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
        self.cars_passing_per_step = 5
        self.car_arrival_dist = [0] * 6 + [1] * 9 + [2] * 6 + [3] * 3 + [4] * 2 + [5] * 1

        # State
        _ = self.reset()

        # Spaces
        self.action_space = gym.spaces.Discrete(self.num_lights + 1)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([self.num_lights + 1, np.inf, np.inf]), dtype=np.float32
        )

    def reset(self, seed: Any = None, options: Any = None) -> tuple[Observation, dict]:
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        self.active_light = np.array([ALL_LIGHTS_OFF])
        self.cars_left = 0
        self.cars_right = 0
        return self._to_state(), {}  # empty info dict

    def step(self, action: Action) -> tuple[Observation, int, bool, bool, dict]:
        if action not in {ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON}:
            raise ValueError(f"Received invalid action={action} which is not part of the action space")

        # Update light
        self.active_light = action

        # Compute reward
        reward = 0
        info = {"passed_left": 0, "passed_right": 0}

        # Let cars pass
        if self.active_light == LEFT_LIGHT_ON:
            passing = min(self.cars_left, self.cars_passing_per_step)
            self.cars_left -= passing
            reward += passing
            info["passed_left"] = passing
        elif self.active_light == RIGHT_LIGHT_ON:
            passing = min(self.cars_right, self.cars_passing_per_step)
            self.cars_right -= passing
            reward += passing
            info["passed_right"] = passing

        # Waiting cars
        reward -= max(self.cars_left - 10, 0)
        reward -= max(self.cars_right - 10, 0)

        # New cars coming
        new_left, new_right = np.random.choice(self.car_arrival_dist), np.random.choice(self.car_arrival_dist)
        info.update(new_left=new_left, new_right=new_right)
        self.cars_left += new_left
        self.cars_right += new_right

        # Finished ?
        terminated = bool(self.cars_left == 0 and self.cars_right == 0)
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
            print(f"Waiting: {self.cars_left} | {self.cars_right} --- Active Light: {self.active_light}")

    def close(self) -> None:
        pass

    def _to_state(self) -> Observation:
        return np.array([self.active_light, self.cars_left, self.cars_right]).astype(np.float32)


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = TrafficLightEnv()
    check_env(env, warn=True)
    print("Env checked successfully")
