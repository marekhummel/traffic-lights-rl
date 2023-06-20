from stable_baselines3.common.env_util import make_vec_env

from agents import A2CAgent, RandomAgent  # noqa
from env import TrafficLightEnv

# Init
print("Init")
env_params = {}
env = TrafficLightEnv(**env_params)
# agent = RandomAgent()
agent = A2CAgent()
vec_env = make_vec_env(TrafficLightEnv, n_envs=1, env_kwargs=env_params, seed=123)


# Reset
print("Reset")
obs = vec_env.reset()

# Train
agent.train(env)

# Run
print("---------------------------")
print("Step #0")
vec_env.render()
print()

max_steps = 20
for step in range(max_steps):
    print(f"Step #{step + 1}")

    # Get action
    action = agent.get_action(obs)  # noqa
    print("Action:", action)

    # Get env response
    obs, reward, done, info = vec_env.step(action)

    # Check if done
    if done:
        print("Goal reached!")
        break

    # Print
    print(f"Response: obs={obs}, reward={reward}, done={done}, info={info}")
    print("State:  ", end="")
    vec_env.render()
    print()
