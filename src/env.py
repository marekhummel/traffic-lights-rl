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

        # State
        _ = self.reset()

        # Spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

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
        passing = np.random.choice([2, 3, 4, 5], p=[0.1, 0.7, 0.15, 0.05])
        if action == LEFT_LIGHT_ON:
            passing = min(len(self.cars_left), passing)
            self.cars_left = self.cars_left[passing:]
            reward += passing
            info["passed_left"] = passing
        elif action == RIGHT_LIGHT_ON:
            passing = min(len(self.cars_right), passing)
            self.cars_right = self.cars_right[passing:]
            reward += passing
            info["passed_right"] = passing

        # Bonus if traffic is zero
        if not self.cars_left and not self.cars_right:
            reward += 10

        # Penalize for cars waiting too long
        if max(self.cars_left or [0]) > 10 or max(self.cars_right or [0]) > 10:
            reward -= max(self.cars_left + self.cars_right)

        # Waiting cars wait time
        self.cars_left = [c + 1 for c in self.cars_left]
        self.cars_right = [c + 1 for c in self.cars_right]

        # New cars coming
        new_left, new_right = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.225, 0.35, 0.225, 0.125, 0.05, 0.025], size=2)
        info.update(new_left=new_left, new_right=new_right)
        self.cars_left.extend([0] * new_left)
        self.cars_right.extend([0] * new_right)

        # Finished ?
        terminated = False  # bool(len(self.cars_left) == 0 and len(self.cars_right) == 0)
        truncated = bool(len(self.cars_left) > 30 or len(self.cars_right) > 30)
        if truncated:
            reward -= 1000

        # Return
        return (
            self._to_state(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        if self.render_mode == "console":
            left = len(self.cars_left)  # " ".join([str(c) for c in reversed(self.cars_left)])
            right = len(self.cars_right)  # " ".join([str(c) for c in self.cars_right])
            print(f"{left} -> |    | <- {right}")

    def close(self) -> None:
        pass

    def _to_state(self) -> Observation:
        def light_state(cars: list[int]) -> tuple[int]:
            # return (len(cars), max(cars), sum(cars), sum(cars) / len(cars)) if cars else (0, 0, 0, 0)
            return (len(cars),) if cars else (0,)

        return np.array(light_state(self.cars_left) + light_state(self.cars_right)).astype(np.float32)


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = TrafficLightEnv()
    check_env(env, warn=True)
    print("Env checked successfully")
