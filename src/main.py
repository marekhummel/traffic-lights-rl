from stable_baselines3.common.env_util import make_vec_env
from matplotlib import pyplot as plt

from agents import A2CAgent, RandomAgent, DQNAgent, PPOAgent  # noqa
from env import TrafficLightEnv
from util import ALL_LIGHTS_OFF, LEFT_LIGHT_ON, RIGHT_LIGHT_ON

# Init
print("Init")

env_params = {}
env = TrafficLightEnv(**env_params)
agent = PPOAgent(env, 20_000)
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

max_steps = 20000
cars = []
actions = []
for step in range(max_steps):
    # Get action
    action = agent.get_action(obs)  # type: ignore
    actions.append(action[0])

    # Get env response
    obs, reward, done, info = vec_env.step(action)

    # Check if done
    if done:
        print(f"Response: done={done}, info={info}")
        print("Goal reached!")
        break

    cars.append((obs[0][0], obs[0][1]))

    # Print
    if step % 1000 == 0:
        print(f"Step #{step + 1}")
        print("Action:", action)
        print(f"Response: obs={obs}, reward={reward}, done={done}, info={info}")
        print("State:  ", end="")
        vec_env.render()
        print()

print(f"Done after {step} steps")

# Plot
obs = info[0]["terminal_observation"]
cars.append((obs[0], obs[1]))

cars_left, cars_right = map(list, zip(*cars))
max_cars = max(cars_left + cars_right)

total_cars = [cl + cr for cl, cr in cars]
cars_left = [-c for c in cars_left]
ACTION_MAP = {ALL_LIGHTS_OFF: 0, LEFT_LIGHT_ON: -max_cars - 1, RIGHT_LIGHT_ON: max_cars + 1}
actions = [ACTION_MAP[a] for a in actions]


figure, axis = plt.subplots(2, figsize=(16, 12), dpi=80)


axis[0].fill_betweenx(list(range(len(cars))), cars_left, cars_right)
axis[0].scatter(actions, list(range(len(actions))), s=1.5, c="black")  # action taken
axis[0].axvline(0, c="black", linewidth=0.4)

ticks = list(range(0, int(max_cars * 1.2) + 1, max(int(max_cars * 1.2) // 5, 1)))
ticks = [-t for t in reversed(ticks)] + ticks
axis[0].set_xticks(ticks)
axis[0].set_title("Cars distribution left/right")
axis[0].set_ylabel("Time step")
axis[0].set_xlabel("#cars (<- left | right ->)")


axis[1].set_title("Total cars")
axis[1].set_xlabel("Time step")
axis[1].set_ylabel("#cars")
axis[1].bar(list(range(len(total_cars))), total_cars, width=1.0, color="orange")

plt.show()
