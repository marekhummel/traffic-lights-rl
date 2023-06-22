from stable_baselines3 import DQN

from agents.agent import Agent
from util import Action, Environment, Observation


class DQNAgent(Agent):
    model: DQN

    def __init__(self, env: Environment, learning_steps: int) -> None:
        self.learning_steps = learning_steps
        self.model = DQN("MlpPolicy", env, verbose=0)

    def _train(self) -> None:
        self.model.learn(self.learning_steps)

    def _get_action(self, obs: Observation) -> Action:
        return self.model.predict(obs, deterministic=True)[0]
