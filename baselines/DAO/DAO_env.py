import pandas as pd
import math
import numpy as np

RANDOM_SEED = 42

FPS = 25
Length = 2  # each segment duration
RTT = 0.08  # 80ms

INFER_T = 0.0025


class Environment:
    def __init__(
            self,
            cooked_bw,
            seq_chunk_data,
            seq_chunks,
            start,
            seq_id,
            random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)

        self.seq_chunk_data = seq_chunk_data
        # self.df = pd.read_hdf(h5_file, 'encoding_data')
        self.SEQ_CHUNKS = seq_chunks
        self.FRAME = []
        self.seq_id = seq_id
        self.video_chunk_counter = 0
        self.cooked_bw = cooked_bw * 2
        #  start point of the trace
        # 表示编码当前段的开始传输时间，初始应该是2，因为得等第一个2s段拍出来，但是我们认为初始状态下已经准备好第一个段，所以初始值为0
        self.start = start
        self.video_start_shoot = self.start - 2  # 用来计算传输延迟
        self.start_shoot_fix = self.video_start_shoot
        self.server_bu = []
        self.last_end = 0

        self.bw = []
        self.transfer_lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance
        self.wait_t = []
        self.bbit = []
        # self.ww = []
        # self.hh = []
        self.rre = []
        self.real_bit = []
        # self.qp = []
        self.F1 = []
        self.bw_use = []
        self.Reward = []

    def get_video_chunk(self, bit, re, encode_t):
        self.bbit.append(bit)
        self.rre.append(re)

        cfg_id = bit * 3 + re
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[(self.seq_id, self.video_chunk_counter, cfg_id)]
        self.bw_use.append(video_chunk_bit / 1000)
        real_b = video_chunk_bit
        self.real_bit.append(real_b)

        # encode_t = ENCODE_T[bit * 12 + 0 * 3 + re]
        end = self.start + encode_t
        self.lag_1.append(encode_t)
        # 当前 chunk 从拍完（采集完成）到被编码之间的等待时间，这个延迟记录的是编码当前chunk是否需要等待
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        bw = 0
        latency = 0
        v_s = video_chunk_size
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
                self.lag_3.append(end - encode_t - self.start + RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw / 125
                self.bw.append(bw)
                self.start = end + RTT
                break

        # 模拟服务器端推理处理的排队机制
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

        latency += wait_t(self.server_bu) + self.FRAME[0] * INFER_T
        self.transfer_lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(self.FRAME[0] * INFER_T)

        self.server_bu.append([end, self.FRAME[0]])

        # 计算精度
        m = video_chunk_acc
        # print("video ",self.seq_id, "chunk ", self.video_chunk_counter, "config (",bit, ",", re, ") accuracy : ",m)
        self.F1.append(m)
        reward = 4 * m - 1.5 * max((latency - 2), 0)
        self.Reward.append(reward)
        # print("chunk ",self.video_chunk_counter, " : ", reward)

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，不需要等待服务器分析上一个块
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        bw_est = self.bw[-1]

        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            self.video_chunk_counter = 0
            end_of_video = True

        return bw_est, self.video_chunk_counter+1, end_of_video



def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else:
        return 0
