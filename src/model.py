from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from env import GoLeftEnv

# Instantiate the env
vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(grid_size=10), seed=123)

# Train the agent
env = GoLeftEnv()
model = A2C("MlpPolicy", env, verbose=1).learn(5000)

# Test the trained agent
# using the vecenv
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)  # type: ignore
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
