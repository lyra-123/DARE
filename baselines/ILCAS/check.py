import torch
import torch.nn as nn
import math
import cv2
import numpy as np
import os
import pandas as pd



class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_dim = 384
        self.mfcnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64, self.motion_dim),  # 映射到 384
            nn.ReLU()
        )

    def forward(self, s):
        B = s.size(0)
        mf_map = s.view(B, 1, 60, 80)
        mf_feat = self.mfcnn(mf_map)
        return mf_feat  # (B, motion_dim)

def resize_image(image, size):
    ih, iw= image.shape
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image_back = np.ones((h, w), dtype=np.uint8) * 128
    image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = image
    return image_back

# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# model = AgentNet()
# gray_uint8 = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
# gray_float = gray_uint8.astype(np.float32)
# gray_float = resize_image(gray_float, (60, 80))
# print(gray_float)
# gray_normalized = gray_float / 255.0  # dtype=float32, 范围 [0.0, 1.0]
# gray_normalized = torch.from_numpy(gray_normalized)
# # state = torch.randn(2, h4 * w4)
# print(model(gray_normalized.unsqueeze(0).type(dtype)).shape)


# df = pd.read_csv(r'D:\Pycharmproject\ILCAS\017.csv')
#
# # 4×4 小块网格尺寸
# S = 4
# H4 = math.ceil(480 / S)
# W4 = math.ceil(854 / S)
#
# total = np.zeros((H4, W4), np.float32)
# # —— 遍历 chunk 内所有宏块
# for _, row in df.iterrows():
#     print("each row ",row.motion_x,",",row.motion_y)
#     deg = (abs(row.motion_x) + abs(row.motion_y)) / row.motion_scale
#     print("each deg ", deg)
#     # print("each deg ",deg)
#     cx, cy, bw, bh = int(row.dstx), int(row.dsty), int(row.blockw), int(row.blockh)
#     x_lt = cx - bw // 2
#     y_lt = cy - bh // 2
#     x_rb = cx + (bw - 1) // 2
#     y_rb = cy + (bh - 1) // 2
#     j0, i0 = x_lt // S, y_lt // S
#     j1, i1 = x_rb // S, y_rb // S
#     for ii in range(i0, i1 + 1):
#         for jj in range(j0, j1 + 1):
#             if 0 <= ii < H4 and 0 <= jj < W4:
#                 total[ii, jj] += deg
#
# # —— 对 25帧求平均 (公式3)
# avg = total / 2
# avg = np.clip(avg, 0, 255)
#
# # —— σ‑拉伸 (公式4)
# gray = np.where(avg >= 20, 255,
#                 avg * (255.0 / 20)).astype(np.uint8)
# out_path = 'output1.png'
# cv2.imwrite(out_path, gray)
# gray = resize_image(gray, (64, 64))
#
# # —— 保存灰度 PNG
# out_path = 'output2.png'
# cv2.imwrite(out_path, gray)
# print(f'✅{out_path}')

# ref_video_dir = '/home/dell/lyra/ILCAS/mv_chunks/LMOT2'
# cur_video_dir = '/home/dell/lyra/ILCAS/mv_chunks/LMOT_dark_rgb_trainval'
# ref_video_file = sorted(os.listdir(ref_video_dir))
# cur_video_file = sorted(os.listdir(cur_video_dir))
#
# for video_file in ref_video_file:
#     ref_video_path = os.path.join(ref_video_dir, video_file)
#     cur_video_path = os.path.join(cur_video_dir, video_file)
#
#     ref_chunk_dir = sorted(os.listdir(ref_video_path))
#     cur_chunk_dir = sorted(os.listdir(cur_video_path))
#
#     for chunk in ref_chunk_dir:
#         ref_chunk_path = os.path.join(ref_video_path, chunk)
#         cur_chunk_path = os.path.join(cur_video_path, chunk)
#
#         ref_image_dir = sorted(os.listdir(ref_chunk_path))
#         cur_image_dir = sorted(os.listdir(cur_chunk_path))
#
#         for image_file in ref_image_dir:
#             ref_image_path = os.path.join(ref_chunk_path, image_file)
#             cur_image_path = os.path.join(cur_chunk_path, image_file)
#             print("check ", ref_image_path)
#
#             img1 = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
#             img2 = cv2.imread(cur_image_path, cv2.IMREAD_GRAYSCALE)
#
#             if not np.array_equal(img1, img2):
#                 print("两幅图像完全一致")
#             # else:
#             #     print("两幅图像存在像素不同")

# video_file = '/home/dell/lyra/CASVA/dataset/video_DETRAC_train'
#
# def get_seq_chunks_list(SEQ):
#     chunk_list = sorted(os.listdir(video_file))
#     seq_chunks = {seq_id: 0 for seq_id in range(SEQ)}
#     for chunk in chunk_list:
#         seq_id = int(chunk.split('_')[0])
#         seq_chunks[seq_id] += 1
#     return seq_chunks
#
# episode_steps = get_seq_chunks_list(27)
# video_dir = '/home/dell/lyra/ILCAS/mv_chunks/DETRAC'
# video_pth = sorted(os.listdir(video_dir))
# for idx, video in enumerate(video_pth):
#     video_folder = os.path.join(video_dir, video)
#     chunk_dir = sorted(os.listdir(video_folder))
#     if not len(chunk_dir)==episode_steps[idx]:
#         print(video_folder," : ",len(chunk_dir), ",", episode_steps[idx])
#     for chunk in chunk_dir:
#         chunk_path = os.path.join(video_folder, chunk)
#         if not len(os.listdir(chunk_path))==120:
#             print(chunk_path)
#
# SEQUENCES = sorted(os.listdir('/home/dell/lyra/Dataset/DETRAC/train/images'))
# print(SEQUENCES)

df = pd.read_csv('/home/dell/lyra/ILCAS/DETRAC/mvs/MVI_63562/012/048.csv', low_memory=False)

# 检查所有列的 NaN 数量
print("每列缺失值统计：")
print(df.isna().sum())

# 检查和你代码相关的几列
print("\n目标列缺失值统计：")
print(df[['dstx', 'dsty', 'blockw', 'blockh']].isna().sum())

# 如果想直接看有 NaN 的行
nan_rows = df[df[['dstx','dsty','blockw','blockh']].isna().any(axis=1)]
print("\n包含 NaN 的行数：", len(nan_rows))
print(nan_rows.head(10))  # 只展示前 10 行

