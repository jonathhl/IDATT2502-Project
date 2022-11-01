from Constants import memory_size

"""
    Buffer to store replays and past experience from runs.

    Input
        Capacity: also commonly named buffer_size, takes the memory_size for the code.
"""
class ReplayMemory(object):
    def __init__(self, capacity=memory_size):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.i = 0

    def push(self, info):
        self.capacity[self.i % self.capacity] = info
        self.i += 1

    def sample(self, num_samples):
        assert num_samples < min(self.i, self.capacity)
        if self.i < self.capacity:
            return sample(self.memory[:self.i], num_samples)
        return sample(self.memory, num_samples)