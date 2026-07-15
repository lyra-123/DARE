import pandas as pd
import math
import numpy as np
import cv2
import os

RANDOM_SEED = 28

Length = 2
RTT = 0.08
Lmax = 0.3

INFER_T = 0.0025
# D²-City_1080p、DETRAC、D²-City_720p
FRAME = [50, 25, 16, 8]
# LMOT
# FRAME = [40, 20, 13, 8]

flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

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
        # self.df = pd.read_hdf(h5_file, 'encoding_data')

        # self.TOTAL = 1800
        # self.CHUNK = 1000
        self.SEQ_ID = 0
        self.SEQ_CHUNKS = 0
        self.FRAME = []

        self.video_chunk_counter = 0
        self.start = np.random.randint(2, len(self.cooked_bw))
        self.video_start_shoot = self.start - 2
        self.start_shoot_fix = self.video_start_shoot

        self.Q = 0
        self.bw = []

        self.lag = []
        self.lag_1 = []  # encode
        self.lag_2 = []  # camera wait
        self.lag_3 = []  # trans
        self.lag_4 = []  # server wait
        self.lag_5 = []  # inferance

        self.F1 = []
        self.Reward = []
        self.server_bu = []
        self.last_end = 0

    def get_video_chunk(self, qp, s, r, encode_t):
        cfg_id = qp * 20 + s * 5 + r
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[
            (self.SEQ_ID, self.video_chunk_counter, cfg_id)]
        # video_chunk_size = self.df.loc[(self.SEQ_ID, self.video_chunk_counter, qp, s, r), 'Size']

        end = self.start + encode_t
        self.lag_1.append(encode_t)
        self.lag_2.append(self.start - self.video_start_shoot - Length)
        v_s = video_chunk_size

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
                self.start = end + RTT
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

        latency += wait_t(self.server_bu) + self.FRAME[s] * INFER_T
        self.lag.append(latency)
        self.lag_4.append(wait_t(self.server_bu))
        self.lag_5.append(self.FRAME[s] * INFER_T)
        self.server_bu.append([end, self.FRAME[s]])

        self.Q = max(self.Q + self.lag_3[-1] - Lmax, 0)
        # f = self.df.loc[(index, qp, s, r), 'Accuracy']
        f = video_chunk_acc
        self.F1.append(f)
        self.Reward.append(4 * f - 1.5 * max(0, latency - 2))
        # flow_feat, mag_feat = extract_flow_from_video_chunk(self.SEQ_ID, self.video_chunk_counter, FPS)

        self.video_chunk_counter += 1
        self.video_start_shoot += Length
        end_of_video = False

        # 更新下个块的理想开始传输时间，不需要等待服务器分析上一个块
        if self.start < self.video_start_shoot + Length:
            self.start = self.video_start_shoot + Length

        if self.video_chunk_counter >= self.SEQ_CHUNKS:
            self.video_chunk_counter = 0
            self.Q = 0
            self.bw = []
            end_of_video = True

        return bw/125, self.lag_3[-1]+self.lag_2[-1], v_s/1000000, f, self.Q, self.Reward[-1], end_of_video

    def get_info(self, qp, s, r, encode_t):
        cfg_id = qp * 20 + s * 5 + r
        video_chunk_size, video_chunk_acc, video_chunk_bit = self.seq_chunk_data[
            (self.SEQ_ID, self.video_chunk_counter, cfg_id)]
        # video_chunk_size = self.df.loc[(self.SEQ_ID, self.video_chunk_counter, qp, s, r), 'Size']

        end = self.start + encode_t

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
                trans_latency = end - encode_t - self.start + RTT
                break

        # 模拟了一个服务器buffer
        l_e = self.last_end
        buffer_copy = self.server_bu.copy()
        while buffer_copy:
            if max(l_e, buffer_copy[0][0]) + buffer_copy[0][1] * INFER_T <= end:
                l_e = max(l_e, buffer_copy[0][0]) + buffer_copy[0][1] * INFER_T
                buffer_copy.pop(0)
            else:
                break
        inf_latency = wait_t(buffer_copy) + self.FRAME[s] * INFER_T

        f = video_chunk_acc

        return trans_latency, inf_latency, f


# def extract_flow(frame1, frame2):
#     """Extract optical flow between two frames using Farneback method"""
#     # Convert to grayscale if needed
#     if len(frame1.shape) == 3:
#         gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     else:
#         gray1 = frame1
#
#     if len(frame2.shape) == 3:
#         gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     else:
#         gray2 = frame2
#
#     # Calculate optical flow
#     flow = cv2.calcOpticalFlowFarneback(
#         gray1, gray2, None,
#         flow_params['pyr_scale'],
#         flow_params['levels'],
#         flow_params['winsize'],
#         flow_params['iterations'],
#         flow_params['poly_n'],
#         flow_params['poly_sigma'],
#         flow_params['flags']
#     )
#
#     return flow
#
#
# def extract_flow_from_video_chunk(seq_id, chunk_no, sample_interval):
#     """Extract optical flow from video chunk at specified intervals"""
#     flows = []
#
#     seq_name = video_dir[seq_id]
#     video_name = os.path.join(image_dir, seq_name)
#     frame_files = sorted([f for f in os.listdir(video_name) if f.endswith((".jpg", ".png"))])
#
#     chunk_start = chunk_no * FRAMES_PER_CHUNK
#     intervals = [(0, sample_interval-1), (sample_interval, sample_interval+sample_interval-1)]
#     for start, end in intervals:  # 两次：1s 和 2s
#         prev_path = os.path.join(video_name, frame_files[chunk_start + start])
#         curr_path = os.path.join(video_name, frame_files[chunk_start + end])
#         prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
#         curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
#         flow = extract_flow(prev, curr)
#         flows.append(flow)
#
#     # Concatenate flows
#     if flows:
#         flow_cat = np.concatenate(flows, axis=2)
#         # 幅值 (H, W, num_flows)
#         mags = [np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).mean() for f in flows]
#         mag_value = float(np.mean(mags))
#     else:
#         # Return zero flow if not enough frames
#         h, w = prev.shape[:2]
#         flow_cat = np.zeros((h, w, 4))  # 假设 num_flows=2 → 4通道
#         mag_value = 0.0
#
#     return flow_cat, mag_value

def wait_t(buffer):
    if buffer:
        t = 0
        for item in buffer:
            t += item[1] * INFER_T
        return t
    else: return 0