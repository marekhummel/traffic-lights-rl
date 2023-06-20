import gymnasium
from numpy import ndarray
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs  # noqa

Action = ndarray
Environment = gymnasium.Env
Observation = ndarray  # VecEnvObs


ALL_LIGHTS_OFF = 0
LEFT_LIGHT_ON = 1
RIGHT_LIGHT_ON = 2
