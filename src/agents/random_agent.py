import numpy as np

from agents.agent import Agent
from util import ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON, Action, Observation


class RandomAgent(Agent):
    def _train(self) -> None:
        pass

    def _get_action(self, obs: Observation) -> Action:
        return np.array([np.random.choice([ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON])])
