import os

import pandas as pd
import math
import numpy as np

RANDOM_SEED = 28

Length = 2
RTT = 0.08
# 模拟分析时间
INFER_T = 0.0025

class Environment:
    def __init__(
        self,
        cooked_trace,
        seq_chunk_data,
        random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)
        self.cooked_bw = cooked_trace
        self.seq_chunk_data = seq_chunk_data

        self.SEQ_ID = 0
        self.SEQ_CHUNKS = 0
        self.FRAME = []
        self.video_chunk_counter = 0
        self.start = np.random.randint(2, len(self.cooked_bw)) # 2139395
        self.video_start_shoot = self.start - 2
        self.start_shoot_fix = self.video_start_shoot

        # 初始选最低配置
        self.last_qp = 4
        self.last_s = 3
        self.last_r = 3

        self.lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance

        self.server_bu = []
        # 这个变量表示的是服务器从哪个时间节点可以开始接新的任务
        self.last_end = 0

    def get_video_chunk(self, qp, s, r, encode_t):
        cfg_id = qp * 20 + s * 5 + r
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[(self.SEQ_ID, self.video_chunk_counter, cfg_id)]

        end = self.start + encode_t
        self.lag_1.append(encode_t)
        # 当前 chunk 从拍完（采集完成）到被编码之间的等待时间，这个延迟记录的是编码当前chunk是否需要等待
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        bw = 0
        latency = 0
        v_s = video_chunk_size
        transmission_start = end

        # 模拟传输，cooked_bw是以秒为单位记录的
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
                self.lag_3.append(end - encode_t - self.start + RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw/125
                transmission_end = end + RTT
                # 如果还有下一个chunk，计算预编码结束时间
                if self.video_chunk_counter + 1 < self.SEQ_CHUNKS:
                    # 预编码在传输期间进行
                    pre_encode_start = transmission_start  # 传输开始时就可以预编码
                    pre_encode_end = pre_encode_start + encode_t
                    # 下一个chunk最早开始时间 = max(传输结束, 预编码结束)
                    self.start = max(transmission_end, pre_encode_end)
                else:
                    # 没有下一个chunk了
                    self.start = transmission_end
                break

        # 模拟了一个服务器buffer
        l_e = self.last_end
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

        # 这里的latency计算的是总延迟，即从拍摄完成到服务器分析结束
        # 在线过程中，输入的upload delay不包含服务器处理时间
        latency += wait_t(self.server_bu) + self.FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(self.FRAME[s] * INFER_T)

        self.server_bu.append([end, self.FRAME[s]])

        # f = self.df.loc[(index, qp, s, r), 'Accuracy']
        f = video_chunk_acc

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，需要判断拍摄好的时间
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        # 计算video content dynamics
        if self.video_chunk_counter < self.SEQ_CHUNKS:
            chunk_size_next, _, _ = self.seq_chunk_data[(self.SEQ_ID, self.video_chunk_counter, cfg_id)]
            dynamics = (chunk_size_next - v_s)/v_s
        else:
            dynamics = 0

        # buffer size
        buffer_size = ((self.start - self.start_shoot_fix)//2 - self.video_chunk_counter) * Length

        # print("check video_chunk_counter: ", self.video_chunk_counter," , ", self.SEQ_CHUNKS[self.seq_counter])
        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            self.video_chunk_counter = 0
            end_of_video = True
        return bw/1000, self.lag_3[-1]+self.lag_2[-1], buffer_size, v_s/1000000, dynamics, f, end_of_video


def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0
