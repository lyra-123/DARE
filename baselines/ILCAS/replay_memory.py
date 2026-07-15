import random
import torch

# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#
#     def push(self, events):
#         for event in zip(*events):
#             self.memory.append(event)
#             if len(self.memory) > self.capacity:
#                 del self.memory[0]
#
#     def clear(self):
#         self.memory = []
#
#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size))
#         # samples = zip(*self.memory[:batch_size])
#         return map(lambda x: torch.cat(x, 0), samples)
#
#     def pop(self, batch_size):
#         mini_batch = zip(*self.memory[:batch_size])
#         return map(lambda x: torch.cat(x, 0), mini_batch)
#
#     def return_size(self):
#         return len(self.memory)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0  # actor连续采样位置

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory) > self.capacity:
                # 如果超过容量，丢弃最旧的数据
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size, continuous=False):
        if len(self.memory) == 0:
            return None

        if continuous:
            # 循环连续采样
            idxes = [(self.pos + i) % len(self.memory) for i in range(batch_size)]
            batch = [self.memory[i] for i in idxes]
            self.pos = (self.pos + batch_size) % len(self.memory)
        else:
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))

        samples = zip(*batch)
        return map(lambda x: torch.cat(x, 0), samples)

    def return_size(self):
        return len(self.memory)
