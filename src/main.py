from stable_baselines3.common.env_util import make_vec_env
from matplotlib import pyplot as plt

from agents import A2CAgent, RandomAgent, DQNAgent, PPOAgent  # noqa
from env import TrafficLightEnv
from util import ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON

# Init
print("Init")

env_params = {}
env = TrafficLightEnv(**env_params)
agent = A2CAgent(env, 50_000)
vec_env = make_vec_env(TrafficLightEnv, n_envs=1, env_kwargs=env_params, seed=123)


# Reset
print("Reset")
obs = vec_env.reset()

# Train
agent.train()

# Run
print("---------------------------")
print("Step #0")
vec_env.render()
print()

max_steps = 2000
cars = []
actions = []
for step in range(max_steps):
    print(f"Step #{step + 1}")

    # Get action
    action = agent.get_action(obs)  # type: ignore
    actions.append(action[0])
    print("Action:", action)

    # Get env response
    obs, reward, done, info = vec_env.step(action)
    cars.append((obs[0][1], obs[0][3]))

    # Check if done
    if done:
        print(f"Response: done={done}, info={info}")
        print("Goal reached!")
        break

    # Print
    print(f"Response: obs={obs}, reward={reward}, done={done}, info={info}")
    print("State:  ", end="")
    vec_env.render()
    print()


# Plot
figure, axis = plt.subplots(2, figsize=(16, 12), dpi=80)

cars_left, cars_right = map(list, zip(*cars))
max_cars = max(cars_left + cars_right)

total_cars = [cl + cr for cl, cr in cars]
cars_left = [-c for c in cars_left]
ACTION_MAP = {ALL_LIGHTS_OFF: 0, LEFT_LIGHT_ON: -max_cars // 5, RIGHT_LIGHT_ON: max_cars // 5}
actions = [ACTION_MAP[a] for a in actions]

axis[0].plot(list(range(len(cars_left))), cars_left)  # position
axis[0].plot(list(range(len(cars_right))), cars_right)  # velocity
axis[0].scatter(list(range(len(actions))), actions, s=0.5, c="black")  # action taken
axis[0].axhline(0, c="gray", linewidth=0.4)

axis[1].plot(list(range(len(total_cars))), total_cars)

plt.show()
