import gymnasium
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from agents.agent import Agent
from util import ACTION_LEFT
import numpy as np


class HardcodedLeftAgent(Agent):
    def _train(self, env: gymnasium.Env) -> None:
        pass

    def _get_action(self, obs: VecEnvObs):
        return np.array([ACTION_LEFT])
