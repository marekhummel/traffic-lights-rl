from stable_baselines3 import A2C

from agents.agent import Agent
from util import Action, Environment, Observation


class A2CAgent(Agent):
    model: A2C

    def _train(self, env: Environment) -> None:
        self.model = A2C("MlpPolicy", env, verbose=0).learn(5000)

    def _get_action(self, obs: Observation) -> Action:
        return self.model.predict(obs, deterministic=True)[0]
