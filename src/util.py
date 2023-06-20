from typing import Literal

import gymnasium
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

Action = Literal[0, 1]
Environment = gymnasium.Env
Observation = VecEnvObs

ACTION_LEFT = 0
ACTION_RIGHT = 1
