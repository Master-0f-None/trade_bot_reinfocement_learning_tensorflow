import gym
import json
import datetime as dt

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
#
# import env
from A2C import A2CAgent
from env import StockTradeEnv
import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')
df = df.drop("index",axis=1)
print(df.head())
print(df.iloc[0])
# The algorithms require a vectorized environment to run
env = StockTradeEnv(df)
# env = gym.make("CartPole-v0")
print(env.action_space)
print(env.observation_space)
MAX_EPISODE = 1500
MAX_STEPS = 500
gamma = 0.99
lr = 0.001
agent = A2CAgent(env,gamma,lr)


def run():
    for episode in range(MAX_EPISODE):
        state = env.reset()
        trajectory = []  # [[s, a, r, s', done], [], ...]
        episode_reward = 0
        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward

            if done:
                break

            state = next_state
        if episode % 10 == 0:
            print("Episode " + str(episode) + ": " + str(episode_reward))
        agent.update(trajectory)


run()
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000)
#
# obs = env.reset()
# for i in range(2000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()