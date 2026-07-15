import pandas as pd
import math
import numpy as np
import csv
import os

RANDOM_SEED = 28

Length = 2  # each segment duration
RTT = 0.08  # 80ms
INFER_T = 0.0025
FRAMES = 50

class Environment:
    def __init__(
            self,
            cooked_bw,
            seq_chunk_data,
            seq_chunks,
            start,
            random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)

        # self.df = pd.read_hdf(h5_file, 'encoding_data')
        self.seq_chunk_data = seq_chunk_data
        self.SEQ_CHUNKS = seq_chunks
        self.seq_id = 0
        self.video_chunk_counter = 0
        self.cooked_bw = cooked_bw * 2
        #  start point of the trace
        # 表示编码当前段的开始传输时间，初始应该是2，因为得等第一个2s段拍出来，但是我们认为初始状态下已经准备好第一个段，所以初始值为0
        self.start = start
        self.video_start_shoot = self.start - 2  # 用来计算传输延迟
        self.start_shoot_fix = self.video_start_shoot
        self.server_bu = []
        self.last_end = 0

        self.F1 = []
        self.lag = []
        self.Reward = []
        self.bw_use = []

    def get_video_chunk(self):
        video_chunk_acc, video_chunk_size, video_chunk_bit = self.seq_chunk_data[self.seq_id][self.video_chunk_counter]
        self.bw_use.append(video_chunk_bit / 1000)
        end = self.start
        # 模拟块从客户端传输到服务器的过程
        while True:
            if math.ceil(end) == end:
                if end + 1 >= len(self.cooked_bw):
                    real_bw = self.cooked_bw[int(end + 1) % len(self.cooked_bw)]
                else:
                    real_bw = self.cooked_bw[int(end + 1)]
                duration = 1

            else:
                if math.ceil(end) >= len(self.cooked_bw):
                    real_bw = self.cooked_bw[math.ceil(end) % len(self.cooked_bw)]
                else:
                    real_bw = self.cooked_bw[math.ceil(end)]
                duration = math.ceil(end) - end

            if video_chunk_size - real_bw * 1000 * duration >= 0:
                video_chunk_size = video_chunk_size - real_bw * 1000 * duration
                end += duration
            else:
                end += video_chunk_size / (real_bw * 1000)
                video_chunk_size = 0
            if video_chunk_size == 0:
                latency = end + RTT - self.video_start_shoot - Length
                transmission_end = end + RTT
                # 如果还有下一个chunk，计算预编码结束时间
                self.start = transmission_end
                break

        # 模拟服务器端推理处理的排队机制
        l_e = self.last_end
        # 先看看在当前块传输到服务器前可以清空服务器上的哪些任务，这取决了当前块到达服务器后需不需要等待
        while self.server_bu:
            if max(l_e, self.server_bu[0][0]) + self.server_bu[0][1] * INFER_T <= end:
                l_e = max(l_e, self.server_bu[0][0]) + self.server_bu[0][1] * INFER_T
                self.server_bu.pop(0)
            else:
                break
        if self.server_bu:
            self.last_end = self.server_bu[0][0]
        else:
            self.last_end = end

        latency += wait_t(self.server_bu) + FRAMES * INFER_T
        self.lag.append(latency)

        self.server_bu.append([end, FRAMES])

        # 计算精度
        f = video_chunk_acc
        self.F1.append(f)

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，不需要等待服务器分析上一个块
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        # reward = 2 * f - 1.5 * max((latency - 2), 0) + 0.5 * max(self.last_buff - buffer_size, 0)
        reward = 4 * f - 1.5 * (max(latency - 2, 0))
        self.Reward.append(reward)  # 奖励具体定义看自己，这里与论文并不一致

        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            self.video_chunk_counter = 0
            end_of_video = True

        return self.lag[-1], f, end_of_video



def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else:
        return 0
