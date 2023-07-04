import numpy as np
from agents.agent import Agent
from util import ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON, Action, Observation


class SimpleAgent(Agent):
    last_action: Action

    def _train(self) -> None:
        self.last_action = RIGHT_LIGHT_ON

    def _get_action(self, obs: Observation) -> Action:
        if self.last_action == LEFT_LIGHT_ON:
            self.last_action = RIGHT_LIGHT_ON
            return np.array([RIGHT_LIGHT_ON])
        elif self.last_action == RIGHT_LIGHT_ON:
            self.last_action = LEFT_LIGHT_ON
            return np.array([LEFT_LIGHT_ON])

        return ALL_LIGHTS_OFF


class SimpleAgentBetter(Agent):
    def _train(self) -> None:
        pass

    def _get_action(self, obs: Observation) -> Action:
        left, right = obs[0][:2]
        return np.array([LEFT_LIGHT_ON if left > right else RIGHT_LIGHT_ON])
