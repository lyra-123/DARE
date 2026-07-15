import random
import torch

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def push_video(self, video):
        """
        video = {
            'states': Tensor [T, ...],
            'actions': Tensor [T],
            'f1s': Tensor [T],
            'seq_ids': Tensor [T],
            'chunk_ids': Tensor [T]
        }
        """
        self.memory.append(video)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # samples = zip(*self.memory[:batch_size])
        return map(lambda x: torch.cat(x, 0), samples)

    def sample_videos_by_chunk_budget(self, batch_size):
        """
        按视频取，直到 chunk 总数 >= batch_size
        """
        videos = []
        total_chunks = 0

        for video in random.sample(self.memory, len(self.memory)):
            videos.append(video)
            total_chunks += video['exp_actions'].size(0)
            if total_chunks >= batch_size:
                break

        return videos

    def pop(self, batch_size):
        mini_batch = zip(*self.memory[:batch_size])
        return map(lambda x: torch.cat(x, 0), mini_batch)

    def return_size(self):
        return len(self.memory)