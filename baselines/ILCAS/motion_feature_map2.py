# -*- coding: utf-8 -*-
"""mv_chunks_gpu.py

改写自原始脚本：
1. 仅把“宏块 → 4×4 网格累加”这一步搬上 GPU（Numba‑CUDA 方案）。
2. 修复了 d_total 未初始化导致的随机值问题，新增零初始化。
3. 若未检测到 GPU，会自动退回纯 CPU 实现（与原脚本等价）。
"""

import os
import math
import cv2
import pandas as pd
import numpy as np
import torch
import time

try:
    from numba import cuda
    _has_cuda = cuda.is_available()
except Exception:
    _has_cuda = False

# ────────────────────────────────────────────
# 参数区
# ────────────────────────────────────────────
CSV_BASE_DIR = '/home/dell/lyra/ILCAS/LMOT/mvs'
FPS      = 20         # 帧率
SIGMA    = 20
OUT_DIR  = '/home/dell/lyra/ILCAS/mv_chunks/LMOT'    # 输出文件夹
os.makedirs(OUT_DIR, exist_ok=True)

# DETRAC
# RE = [[960, 540], [768, 432], [576, 324], [480, 270], [384, 216]]
# LMOT
RE = [[1800, 1000], [1440, 800], [1080, 600], [900, 500], [720, 400]]
# D²-City_720p
# RE = [[1280, 720], [1024, 576], [768, 432], [640, 360], [512, 288]]
# DSEC
# RE = [[1440, 1080], [1152, 864], [864, 648], [720, 540], [576, 432]]

# ILCAS
# SKIP = [0, 1, 2, 5, 11]
# ILCAS
# SKIP = [0, 1, 2, 4, 9]
SKIP = [0, 1, 2, 4]

S = 4   # 4×4 小网格

print(f"▶ CUDA support: {_has_cuda}")

# ────────────────────────────────────────────
# Numba CUDA kernel
# ────────────────────────────────────────────
if _has_cuda:
    @cuda.jit
    def accumulate_kernel(dstx, dsty, bw, bh, deg, total, S, H4, W4):
        k = cuda.grid(1)
        if k >= dstx.size:
            return

        cx = dstx[k]
        cy = dsty[k]
        w  = bw[k]
        h  = bh[k]
        d  = deg[k]

        x_lt = cx - w // 2
        y_lt = cy - h // 2
        x_rb = cx + (w - 1) // 2
        y_rb = cy + (h - 1) // 2

        j0 = x_lt // S
        i0 = y_lt // S
        j1 = x_rb // S
        i1 = y_rb // S

        for ii in range(i0, i1 + 1):
            if ii < 0 or ii >= H4:
                continue
            for jj in range(j0, j1 + 1):
                if 0 <= jj < W4:
                    cuda.atomic.add(total, (ii, jj), d)

# ────────────────────────────────────────────
# Fallback CPU 版本
# ────────────────────────────────────────────

def accumulate_cpu(df: pd.DataFrame, H4: int, W4: int) -> np.ndarray:
    total = np.zeros((H4, W4), dtype=np.float32)
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
    return total

# ────────────────────────────────────────────
# GPU 版本包装
# ────────────────────────────────────────────

def accumulate_gpu(df: pd.DataFrame, H4: int, W4: int) -> np.ndarray:
    # 提取列并转换 dtype
    dstx  = df['dstx'].to_numpy(np.int32)
    dsty  = df['dsty'].to_numpy(np.int32)
    bw    = df['blockw'].to_numpy(np.int32)
    bh    = df['blockh'].to_numpy(np.int32)
    deg   = ((df['motion_x'].abs() + df['motion_y'].abs()) / df['motion_scale']).to_numpy(np.float32)

    # 传数据到 GPU
    d_dstx = cuda.to_device(dstx)
    d_dsty = cuda.to_device(dsty)
    d_bw   = cuda.to_device(bw)
    d_bh   = cuda.to_device(bh)
    d_deg  = cuda.to_device(deg)

    # 初始化全零的 total 数组
    h_total = np.zeros((H4, W4), dtype=np.float32)
    d_total = cuda.to_device(h_total)
    # 或者用以下两行替代：
    # d_total = cuda.device_array((H4, W4), dtype=np.float32)
    # d_total[:] = 0

    threads = 256
    blocks  = (dstx.size + threads - 1) // threads

    start_evt = cuda.event()
    end_evt   = cuda.event()
    start_evt.record()
    accumulate_kernel[blocks, threads](d_dstx, d_dsty, d_bw, d_bh, d_deg, d_total, S, H4, W4)
    end_evt.record(); end_evt.synchronize()
    print(f"    GPU kernel: {cuda.event_elapsed_time(start_evt, end_evt):.3f} ms")

    return d_total.copy_to_host()

# ────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────
VIDEO_DIR = sorted(os.listdir(CSV_BASE_DIR))[-4:-3]
for video in VIDEO_DIR:
    video_path = os.path.join(CSV_BASE_DIR, video)
    chunk_dir = sorted(os.listdir(video_path))
    for chunk in chunk_dir:
        # if int(chunk) <= 21:
        #     continue
        chunk_path = os.path.join(video_path, chunk)
        config_dir = sorted(os.listdir(chunk_path))
        output_dir = os.path.join(OUT_DIR, video, chunk)
        os.makedirs(output_dir, exist_ok=True)
        for config in config_dir:
            start_total = time.time()

            csv_path = os.path.join(chunk_path, config)
            df = pd.read_csv(csv_path)
            file_name = os.path.splitext(config)[0]
            knob = int(file_name)

            remainder = knob % 20
            skip_idx  = remainder // 5
            re_idx    = remainder % 5
            H4 = math.ceil(RE[re_idx][1] / S)
            W4 = math.ceil(RE[re_idx][0] / S)
            FRAMES = FPS // (SKIP[skip_idx] + 1)

            # —— 核心累加 ——
            if _has_cuda:
                total = accumulate_gpu(df, H4, W4)
            else:
                total = accumulate_cpu(df, H4, W4)

            # —— 后处理 ——
            avg  = np.clip(total / FRAMES, 0, 255)
            gray = np.where(avg >= SIGMA, 255,
                            avg * (255.0 / SIGMA)).astype(np.uint8)

            out_path = os.path.join(output_dir, f"{file_name}.jpg")
            cv2.imwrite(out_path, gray)

            elapsed = time.time() - start_total
            print(f"✅ {video}_{chunk} → {out_path}  ⏱ {elapsed:.2f}s")

print("全部完成！")
