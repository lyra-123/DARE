import cv2
import numpy as np
import os
import pandas as pd
import h5py

from flow_mag_loader import FlowMagLoader
from utils import get_seq_chunks_list_by_h5
# # 读两帧并转灰度
# frame1 = cv2.imread("/home/dell/lyra/Dataset/DETRAC/train/images/MVI_39271/0000.jpg")
# frame2 = cv2.imread("/home/dell/lyra/Dataset/DETRAC/train/images/MVI_39271/0025.jpg")
# prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
# # 计算光流
# flow = cv2.calcOpticalFlowFarneback(
#     prev, next, None,
#     pyr_scale=0.5, levels=3, winsize=15,
#     iterations=3, poly_n=5, poly_sigma=1.2,
#     flags=0
# )
#
# # 光流可视化
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# hsv[..., 0] = ang * 180 / np.pi / 2  # 方向
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 速度
# rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
# cv2.imwrite("flow.png", rgb)

image_dir = '/home/dell/lyra/Dataset/LMOT_dark_rgb_trainval/train/images'
video_dir = sorted(os.listdir(image_dir))
h5_file = '/home/dell/lyra/CASVA/h5_file/LMOT_train.h5'
CHUNK_FRAMES = 40

flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
def extract_flow(frame1, frame2):
    """Extract optical flow between two frames using Farneback method"""
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        flow_params['pyr_scale'],
        flow_params['levels'],
        flow_params['winsize'],
        flow_params['iterations'],
        flow_params['poly_n'],
        flow_params['poly_sigma'],
        flow_params['flags']
    )

    return flow


def extract_flow_from_video_chunk(seq_id, chunk_no, sample_interval):
    """Extract optical flow from video chunk at specified intervals"""
    flows = []

    seq_name = video_dir[seq_id]
    video_name = os.path.join(image_dir, seq_name)
    frame_files = sorted([f for f in os.listdir(video_name) if f.endswith((".jpg", ".png"))])

    chunk_start = chunk_no * CHUNK_FRAMES
    intervals = [(0, sample_interval-1), (sample_interval, sample_interval+sample_interval-1)]
    for start, end in intervals:  # 两次：1s 和 2s
        prev_path = os.path.join(video_name, frame_files[chunk_start + start])
        curr_path = os.path.join(video_name, frame_files[chunk_start + end])
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
        flow = extract_flow(prev, curr)
        flows.append(flow)

    # Concatenate flows
    if flows:
        flow_cat = np.concatenate(flows, axis=2)
        # 幅值 (H, W, num_flows)
        mags = [np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).mean() for f in flows]
        mag_value = float(np.mean(mags))
    else:
        # Return zero flow if not enough frames
        h, w = prev.shape[:2]
        flow_cat = np.zeros((h, w, 4))  # 假设 num_flows=2 → 4通道
        mag_value = 0.0

    return flow_cat, mag_value


def flow_to_rgb_uint8(flow_all):
    """
    将4通道光流转为归一化后的伪RGB图像 (uint8, 0-255)
    flow_all: numpy array (H, W, 4)
    return: numpy array (H, W, 3), dtype uint8
    """
    # step1: 对通道取均值 (H,W)
    flow_gray = np.mean(flow_all, axis=2).astype(np.float32)

    # step2: 归一化到 0-255
    flow_gray = cv2.normalize(flow_gray, None, 0, 255, cv2.NORM_MINMAX)

    # step3: 转伪RGB (H,W,3)
    flow_rgb = cv2.cvtColor(flow_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return flow_rgb


# 假设这两个是你的参数
total_seq = 11              # 例如有 2 个序列
seq_chunks_list = get_seq_chunks_list_by_h5(h5_file)  # 每个序列对应多少 chunk
interval = 20        # 假设采样间隔

# 输出文件
flow_h5_path = "/home/dell/lyra/LCA/flow_mag/flows_LMOT.h5"
mag_h5_path = "/home/dell/lyra/LCA/flow_mag/mags_LMOT.h5"

# # 使用
# loader = FlowMagLoader(flow_h5_path, mag_h5_path)
# flow = loader.get_flow(0, 10)   # seq 0, chunk 10
# mag = loader.get_mag(0, 10)
# flow_cat_new, mag_value_new = extract_flow_from_video_chunk(0, 10, interval)
#
# diff_flow = np.abs(flow_cat_new - flow).mean()
# diff_mag = abs(mag_value_new - mag)
# print(f"[Seq {0}, Chunk {10}] flow_diff={diff_flow:.6f}, mag_diff={diff_mag:.6f}")

# ========== 保存 ==========
# 光流用 h5py
flow_h5 = h5py.File(flow_h5_path, "w")

# 幅值用 pandas
mag_records = []

for seq in range(total_seq):
    for chunk_id in range(seq_chunks_list[seq]):
        flow_all, mag_value = extract_flow_from_video_chunk(seq, chunk_id, interval)

        # 转为归一化伪RGB (uint8)
        flow_rgb = flow_to_rgb_uint8(flow_all)

        # 存光流 (压缩+节省空间)
        flow_h5.create_dataset(
            f"{seq}/{chunk_id}",
            data=flow_rgb,
            compression="gzip",
            compression_opts=4
        )

        # 存幅值
        mag_records.append({
            "seq_id": seq,
            "chunk_id": chunk_id,
            "mag_value": mag_value
        })

flow_h5.close()

df_mag = pd.DataFrame(mag_records)
df_mag.to_hdf(mag_h5_path, key="mag", mode="w")

print("✅ 数据保存完成！")

# ========== 读取并校验 ==========
flow_h5 = h5py.File(flow_h5_path, "r")
df_mag_loaded = pd.read_hdf(mag_h5_path, key="mag")

for seq in range(total_seq):
    for chunk_id in range(seq_chunks_list[seq]):
        # 读取光流 (伪RGB, uint8)
        flow_loaded = flow_h5[f"{seq}/{chunk_id}"][:]

        # 读取幅值
        mag_loaded = df_mag_loaded.query("seq_id==@seq and chunk_id==@chunk_id")["mag_value"].iloc[0]

        # 重新计算一次做对比（注意：要用同样的归一化方法）
        flow_cat_new, mag_value_new = extract_flow_from_video_chunk(seq, chunk_id, interval)
        flow_rgb_new = flow_to_rgb_uint8(flow_cat_new)

        diff_flow = np.abs(flow_rgb_new.astype(np.float32) - flow_loaded.astype(np.float32)).mean()
        diff_mag = abs(mag_value_new - mag_loaded)

        if diff_flow > 0 or diff_mag > 0:
            print(f"[Seq {seq}, Chunk {chunk_id}] flow_diff={diff_flow:.6f}, mag_diff={diff_mag:.6f}")

flow_h5.close()

# def verify_loader(loader, num_samples=10):
#     for i in range(num_samples):
#         # 随机选一个 seq 和 chunk
#         seq_id = np.random.choice(list(set(k[0] for k in loader.flows.keys())))
#         chunks = [k[1] for k in loader.flows.keys() if k[0] == seq_id]
#         chunk_id = np.random.choice(chunks)
#
#         # 从 loader 读
#         flow_saved = loader.get_flow(seq_id, chunk_id, normalize=False)  # (H,W,3), uint8
#         mag_saved  = loader.get_mag(seq_id, chunk_id)
#
#         # 实时提取
#         flow_cat_new, mag_new = extract_flow_from_video_chunk(seq_id, chunk_id, interval)
#         flow_rgb_new = flow_to_rgb_uint8(flow_cat_new)
#
#         # 计算差异
#         diff_flow = np.abs(flow_rgb_new.astype(np.float32) - flow_saved.astype(np.float32)).mean()
#         diff_mag = abs(mag_new - mag_saved)
#
#         print(f"[Seq {seq_id}, Chunk {chunk_id}] flow_diff={diff_flow:.6f}, mag_diff={diff_mag:.6f}")
#
# loader = FlowMagLoader(flow_h5_path, mag_h5_path)
# verify_loader(loader, num_samples=20)
