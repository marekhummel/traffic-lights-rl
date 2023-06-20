import gymnasium
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from agents.agent import Agent


class A2CAgent(Agent):
    model: A2C

    def _train(self, env: gymnasium.Env) -> None:
        self.model = A2C("MlpPolicy", env, verbose=0).learn(5000)

    def _get_action(self, obs: VecEnvObs):
        return self.model.predict(obs, deterministic=True)[0]
