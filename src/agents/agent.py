import gymnasium
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class Agent:
    _is_trained: bool = False

    def train(self, env: gymnasium.Env) -> None:
        assert not self._is_trained
        print("Agent training...")
        self._train(env)
        self._is_trained = True

    def get_action(self, obs: VecEnvObs):
        assert self._is_trained
        return self._get_action(obs)

    def _train(self, env: gymnasium.Env) -> None:
        raise NotImplementedError("Subclass implementation missing")

    def _get_action(self, obs: VecEnvObs):
        raise NotImplementedError("Subclass implementation missing")
