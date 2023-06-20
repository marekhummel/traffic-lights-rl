from util import Action, Environment, Observation


class Agent:
    _is_trained: bool = False

    def train(self, env: Environment) -> None:
        assert not self._is_trained
        print("Agent training...")
        self._train(env)
        self._is_trained = True

    def get_action(self, obs: Observation) -> Action:
        assert self._is_trained
        return self._get_action(obs)

    def _train(self, env: Environment) -> None:
        raise NotImplementedError("Subclass implementation missing")

    def _get_action(self, obs: Observation) -> Action:
        raise NotImplementedError("Subclass implementation missing")
