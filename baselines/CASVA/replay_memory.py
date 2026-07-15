import random
import torch

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.video_memory = {}

    def push(self, events, video_id=None):
        if video_id is not None:
            # 按视频分组存储
            if video_id not in self.video_memory:
                self.video_memory[video_id] = []
            # events: [states, actions, returns, advantages]
            # 每个都是一个episode的数据列表
            episode_data = []
            for event in zip(*events):
                episode_data.append(event)
            self.video_memory[video_id].extend(episode_data)
        else:
            # 原来的方式
            for event in zip(*events):
                self.memory.append(event)
                if len(self.memory) > self.capacity:
                    del self.memory[0]
        # for event in zip(*events):
        #     self.memory.append(event)
        #     if len(self.memory) > self.capacity:
        #         del self.memory[0]

    def clear(self):
        self.memory = []
        self.video_memory = {}

    def get_video_ids(self):
        """返回所有视频ID"""
        return list(self.video_memory.keys())

    def sample_by_video(self, video_id, batch_size):
        if video_id not in self.video_memory:
            return None
        video_data = self.video_memory[video_id]
        if len(video_data) == 0:
            return None
        # 如果数据不足batch_size，就返回全部
        sample_size = min(batch_size, len(video_data))

        samples = zip(*random.sample(video_data, sample_size))
        return map(lambda x: torch.cat(x, 0), samples)

    def get_all_video_data(self, video_id):
        """
        获取指定视频的所有数据（不采样）
        """
        if video_id not in self.video_memory:
            return None
        video_data = self.video_memory[video_id]
        if len(video_data) == 0:
            return None
        samples = zip(*video_data)
        return map(lambda x: torch.cat(x, 0), samples)

    def get_video_size(self, video_id):
        """返回指定视频的样本数量"""
        if video_id not in self.video_memory:
            return 0
        return len(self.video_memory[video_id])

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

    def pop(self, batch_size):
        mini_batch = zip(*self.memory[:batch_size])
        return map(lambda x: torch.cat(x, 0), mini_batch)

    def return_size(self):
        return len(self.memory)