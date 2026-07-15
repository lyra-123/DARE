#!/usr/bin/env python3
# build_motion_feature_map.py
# ------------------------------------------------------------
# 目标：依论文步骤生成最终 2‑D motion feature map
#   1. 逐帧 → 4×4 网格 (H/4 × W/4)
#   2. 跨帧同位置累加并取平均 (公式3)
#   3. clip 到 0‑255
#   4. 按 σ 阈值再做一次线性拉伸 (公式4)
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import cv2
import os
import math
import resource
import platform
import time
import torch

# ────────────────────────────────────────────
# 参数区
# ────────────────────────────────────────────
CSV_BASE_DIR = r'/home/dell/lyra/ILCAS/D2CITY/mvs'
FPS = 25  # 帧率
SIGMA = 20
OUT_DIR = '/home/dell/lyra/ILCAS/mv_chunks/D²-City/1280x720'  # 输出文件夹
os.makedirs(OUT_DIR, exist_ok=True)

# DETRAC
# RE = [[960, 540], [854, 480], [640, 360], [426, 240]]
# D²-City_1080p
# RE = [[1920, 1080], [1280, 720], [960, 540], [720, 480], [320, 240]]
# LMOT
# RE = [[1800, 1000], [1296, 720], [1080, 600], [864, 480]]
# D²-City_720p
RE = [[1280, 720], [960, 540], [854, 480], [426, 240]]
# DSEC
# RE = [[1440, 1080], [1080, 810], [960, 720], [720, 540], [480, 360]]

# ILCAS
SKIP = [0, 1, 2, 5, 11]
# ILCAS
# SKIP = [0, 1, 2, 4, 9]


# ===============================
# 内存函数
# ===============================
def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == 'Darwin':
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def get_current_memory_mb():
    status_path = '/proc/self/status'
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    return get_peak_memory_mb()
# ===============================
# 内存 before
# ===============================
mem_before = get_current_memory_mb()
peak_before = get_peak_memory_mb()
latencies = []

VIDEO_DIR = sorted(os.listdir(CSV_BASE_DIR))
for video in VIDEO_DIR:
    video_path = os.path.join(CSV_BASE_DIR, video)
    chunk_dir = sorted(os.listdir(video_path))
    for chunk in chunk_dir:
        chunk_path = os.path.join(video_path, chunk)
        config_dir = sorted(os.listdir(chunk_path))
        output_dir = os.path.join(OUT_DIR, video, chunk)
        os.makedirs(output_dir, exist_ok=True)
        for config in config_dir:
            t0 = time.perf_counter()
            # ────────────────────────────────────────────
            # 1. 读取 CSV
            # ────────────────────────────────────────────
            config_csv_path = os.path.join(chunk_path, config)
            df = pd.read_csv(config_csv_path)
            file_name = config.split('.')[0]
            knob = int(file_name)

            # 4×4 小块网格尺寸
            S = 4
            remainder = knob % 20
            skip = remainder // 4  # 0 to 3
            re = remainder % 4
            H4 = math.ceil(RE[re][1] / S)
            W4 = math.ceil(RE[re][0] / S)
            FRAMES = FPS // (SKIP[skip] + 1)
            if FRAMES != 25:
                continue
            print("current resolution config (", W4, "x", H4, ")")
            print("current frames config ", FRAMES)

            total = np.zeros((H4, W4), np.float32)
            # —— 遍历 chunk 内所有宏块
            for _, row in df.iterrows():
                deg = (abs(row.motion_x) + abs(row.motion_y)) / row.motion_scale
                cx, cy, bw, bh = int(row.dstx), int(row.dsty), int(row.blockw), int(row.blockh)
                x_lt = cx - bw // 2
                y_lt = cy - bh // 2
                x_rb = cx + (bw - 1) // 2
                y_rb = cy + (bh - 1) // 2
                j0, i0 = x_lt // S, y_lt // S
                j1, i1 = x_rb // S, y_rb // S
                for ii in range(i0, i1 + 1):
                    for jj in range(j0, j1 + 1):
                        if 0 <= ii < H4 and 0 <= jj < W4:
                            total[ii, jj] += deg

            # —— 对 25帧求平均 (公式3)
            avg = total / FRAMES
            avg = np.clip(avg, 0, 255)

            # —— σ‑拉伸 (公式4)
            gray = np.where(avg >= SIGMA, 255,
                            avg * (255.0 / SIGMA)).astype(np.uint8)

            # —— 保存灰度 PNG
            out_path = os.path.join(output_dir, f'{file_name}.jpg')
            cv2.imwrite(out_path, gray)
            print(f'✅ {video}_{chunk}  →  {out_path}')

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

print('全部完成！')

# ===============================
# 8. 输出统计结果
# ===============================
latencies = torch.tensor(latencies)
print("\n【推理时间】")
print(f"  mean: {latencies.mean():.3f} ms")
print(f"  P50:  {latencies.median():.3f} ms")
print(f"  P95:  {latencies.quantile(0.95):.3f} ms")
print(f"  P99:  {latencies.quantile(0.99):.3f} ms")
print("\n【吞吐量】")

# ===============================
# 7. 内存 after
# ===============================
mem_after = get_current_memory_mb()
peak_after = get_peak_memory_mb()

print("\n【空间复杂度（进程法）】")
print(
    f"  current: {mem_before:.1f} MB → {mem_after:.1f} MB "
    f"(+{mem_after - mem_before:.1f} MB)"
)
print(
    f"  peak:    {peak_before:.1f} MB → {peak_after:.1f} MB "
    f"(+{peak_after - peak_before:.1f} MB)"
)

print("\n【GPU 显存】")
print(
    f"  peak allocated: "
    f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB"
)
print(
    f"  peak reserved:  "
    f"{torch.cuda.max_memory_reserved() / 1024 / 1024:.1f} MB"
)
