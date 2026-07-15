import pandas as pd
import math
import numpy as np
import cv2
import os

RANDOM_SEED = 28
Length = 2  # each segment duration
RTT = 0.08  # 80ms

INFER_T = 0.0025


class Environment:
    def __init__(
            self,
            cooked_bw,
            seq_chunk_data,
            start,
            seq_chunks=0,
            seq_id=0,
            random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)

        self.seq_chunk_data = seq_chunk_data
        # self.df = pd.read_hdf(h5_file, 'encoding_data')
        self.SEQ_CHUNKS = seq_chunks
        self.FRAME = []
        self.seq_id = seq_id
        self.video_chunk_counter = 0
        self.last_buff = 0
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
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance
        self.bw_use = []
        self.Reward = []

    def get_video_chunk(self, qp, s, r, encode_t):
        cfg_id = qp * 20 + s * 5 + r
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[(self.seq_id, self.video_chunk_counter, cfg_id)]
        self.bw_use.append(video_chunk_bit / 1000)

        end = self.start + encode_t
        self.lag_1.append(encode_t)
        # 当前 chunk 从拍完（采集完成）到被编码之间的等待时间，这个延迟记录的是编码当前chunk是否需要等待
        self.lag_2.append(self.start - self.video_start_shoot - Length)
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
                # print(self.video_start_shoot + Length, end+RTT, latency)
                self.lag_3.append(end - encode_t - self.start + RTT)
                bw = v_s / (self.lag_3[-1])
                bw = bw / 125
                self.start = end + RTT
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

        latency += wait_t(self.server_bu) + self.FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(self.FRAME[s] * INFER_T)

        self.server_bu.append([end, self.FRAME[s]])

        # 计算精度
        f = video_chunk_acc
        self.F1.append(f)
        reward = 4 * f - 1.5 * max((latency - 2), 0)
        self.Reward.append(reward)  # 奖励具体定义看自己，这里与论文并不一致

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，不需要等待服务器分析上一个块
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        buffer_size = ((self.start - self.start_shoot_fix) // 2 - self.video_chunk_counter) * Length

        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            self.video_chunk_counter = 0
            end_of_video = True

        # index = qp * 16 + s * 4 + r
        # feature_map = get_feature_map(self.seq_id, self.video_chunk_counter, index)

        return bw / 1000, self.lag_3[-1] + self.lag_2[-1], buffer_size, v_s / 1000000, f, end_of_video



def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else:
        return 0

def resize_image(image, size):
    ih, iw= image.shape
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image_back = np.ones((h, w), dtype=np.float32) * 128
    image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = image
    return image_back

def get_feature_map(seq_id,chunk_id, idx, feature_map_dir):
    video_dir = sorted(os.listdir(feature_map_dir))
    seq_name = video_dir[seq_id]
    map_path = os.path.join(feature_map_dir, seq_name,f'{chunk_id:03d}',f'{idx:03d}.jpg')
    gray_uint8 = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    gray_float = gray_uint8.astype(np.float32)
    gray_float = resize_image(gray_float, (36, 36))
    gray_normalized = gray_float / 255.0  # dtype=float32, 范围 [0.0, 1.0]
    return gray_normalized
