import numpy as np
import torch
import cv2
import random
from collections import deque

from Constants import GAMMA, LEARNING_RATE, EPSILON_DECAY, EPSILON_MIN, EPSILON, MAX_MEMORY_LEN, MIN_MEMORY_LEN, \
    BATCH_SIZE, DEVICE
from DQNModel import DQN


class Agent:
    def __init__(self, env):
        self.state_size_h = env.observation_space.shape[0]
        self.state_size_w = env.observation_space.shape[1]
        self.state_size_c = env.observation_space.shape[2]
        self.action_size = env.action_space.n

        self.target_h = 84  # Height after process
        self.target_w = 84  # Width after process
        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20px from top
        self.gamma = GAMMA  # Discount coefficient
        self.lr = LEARNING_RATE
        self.epsilon = EPSILON
        self.decay = EPSILON_DECAY
        self.epsilon_minimum = EPSILON_MIN
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        self.target = DQN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.target.parameters(), lr=self.lr)

    def preProcess(self, frame):
        if frame.size == 210 * 160 * 3:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = image[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Crops 20px from top
            image = cv2.resize(image, (self.target_w, self.target_h))
            image = image.reshape(self.target_w, self.target_h) / 255  # Normalizing
            return image
        else:
            assert False, "Unknown resolution"

    def act(self, state):
        act_protocol = 'Explore' if self.epsilon >= random.uniform(0, 1) else 'Exploit'
        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.target.forward(state)
                action = torch.argmax(q_values).item()
        return action

    def train(self):
        if len(self.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        state_q_values = self.target(state)
        next_states_q_values = self.target(next_state)
        next_states_target_q_values = self.target(next_state)

        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(
            1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.decay
