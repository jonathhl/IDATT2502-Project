import Constants as c
import torch
from Replay import ReplayMemory
from DQNModel import DQN

class DQNAgent:
    def __init__(
        self,
        env,
        input_shape,
        action_shape,
        memory_size=c.memory_size,
        lr=c.lr, epsilon=c.epsilon,
        epsilon_decay=c.epsilon_decay,
        epsilon_min=c.epsilon_min
    ):
        self.env = env
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.sample_size = c.sample_size
        self.memory_size = memory_size
        self.memory = ReplayMemory(input_shape, memory_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DQN(input_shape, action_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, eps=.0001)
