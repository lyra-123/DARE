import pandas as pd
import math
import numpy as np
import csv
from utils import C_R

RANDOM_SEED = 28
Lmax = 0.5
Length = 2
RTT = 0.08
BW_estimate = 5 # 5就够了吗，不用8？
INFER_T = 0.0025

class Environment:
    def __init__(
        self,
        cooked_bw,
        seq_chunk_data,
        start,
        seq_chunks,
        seq_id,
        random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)

        # self.df = pd.read_hdf(h5_file, 'encoding_data')
        self.SEQ_CHUNKS = seq_chunks
        self.FRAME = []
        self.seq_id = seq_id
        self.video_chunk_counter = 0
        self.last_buff = 0
        self.cooked_bw = cooked_bw*2
        self.seq_chunk_data = seq_chunk_data
        #  start point of the trace
        self.start = start
        self.video_start_shoot = self.start - 2  # 用来计算传输延迟
        self.start_shoot_fix = self.video_start_shoot
        self.server_bu = []
        self.last_end = 0

        self.F1 = []
        self.lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans (+RTT)
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance
        self.Reward = []
        self.bw = []
        self.bw_use = []
        self.Q = 0

    def get_video_chunk(self, qp, s, r, encode_t):
        cfg_id = qp * 20 + s * 5 + r
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[(self.seq_id, self.video_chunk_counter, cfg_id)]
        self.bw_use.append(video_chunk_bit / 1000)

        end = self.start + encode_t
        self.lag_1.append(encode_t)
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        bw = 0
        latency = 0
        v_s = video_chunk_size
        transmission_start = end
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
                bw = bw/1000
                self.bw.append(bw)
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

        latency = latency + wait_t(self.server_bu) + self.FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(self.FRAME[s] * INFER_T)

        self.server_bu.append([end, self.FRAME[s]])

        bw_est = bandwidth_predictor(self.bw, BW_estimate)
        self.Q = max(self.Q + self.lag_3[-1] - Lmax, 0)
        # f = self.df.loc[(self.seq_id, self.video_chunk_counter, qp, s, r), 'Accuracy']
        f = video_chunk_acc
        self.F1.append(f)
        # self.Reward.append(2*f-latency)

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，不需要等待服务器分析上一个块
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        if self.video_chunk_counter < self.SEQ_CHUNKS:
            chunk_size_next, _, _ = self.seq_chunk_data[(self.seq_id, self.video_chunk_counter, cfg_id)]
            dynamics = (chunk_size_next - v_s) / v_s
        else:
            dynamics = 0

        buffer_size = ((self.start - self.start_shoot_fix)//2 - self.video_chunk_counter) * Length
        # reward = 2 * f - 1.5 * max((latency - 2), 0) + 0.5 * max(self.last_buff - buffer_size, 0)
        reward = 4 * f - 1.5 * (max(latency - 2, 0))
        self.Reward.append(reward)  # 奖励具体定义看自己，这里与论文并不一致
        self.last_buff = buffer_size

        # if self.video_chunk_counter > self.TOTAL:
        #     end_of_video = True
        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            end_of_video = True

        return bw_est, bw/125, self.lag_3[-1]+self.lag_2[-1], buffer_size, v_s/1000000, dynamics, f, self.Q, end_of_video


def bandwidth_predictor(bw, est):
    bw_subset = bw[-est:] if est <= len(bw) else bw
    r = len(bw_subset) / sum(1 / x for x in bw_subset)
    return r

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0