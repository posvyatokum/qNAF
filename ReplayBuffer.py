import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size, random_seed=42):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        batch = np.random.choice(self.count, batch_size, replace=True)

        s_batch = np.array([self.buffer[_][0] for _ in batch])
        a_batch = np.array([self.buffer[_][1] for _ in batch])
        r_batch = np.array([self.buffer[_][2] for _ in batch])
        t_batch = np.array([self.buffer[_][3] for _ in batch])
        s2_batch = np.array([self.buffer[_][4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
