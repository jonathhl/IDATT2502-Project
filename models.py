import gym
import random
import numpy as np
import torch.nn as nn
from collections import deque


def DQN(input_shape, action_space):
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, padding=2),
        nn.ReLU(),
    )
    return model


class DQNAgent:
    def __init__(self):
        self.env = gym.make('ALE/Assault-v5', render_mode='human')
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.n_episodes = 100
        self.memory = deque(maxlen=2000)
        self.discount = 0.95
        self.epsilon = 1.0
        self.epsilon_min = .001
        self.epsilon_decay = .999
        self.batch_size = 64
        self.train_start = 1000
        self.model = DQN(input_shape=(self.state_size,),
                         action_space=self.action_size)

    # Saves the memory to memory buffer
    def save_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + \
                                       self.discount * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def save(self, name):
        self.model.save(name)

    # def load(self, name):
    #    self.model = load_model(name)

    def run(self, render=False):
        for e in range(self.n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.save_memory(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.n_episodes, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        return
                self.replay()


agent = DQNAgent()
agent.run()
