import gym
import time, math, random
import numpy as np
import torch
from collections import deque

env = gym.make('ALE/Assault-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def random_play():
    score = 0
    env.reset()
    while True:
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            env.close()
            print("Your Score at end of game is: ", score)
            break


random_play()

