import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import math
import time
import csv
from pathlib import Path

from numpy.polynomial.tests.test_classes import classes
from ultralytics import YOLO
import cv2
from utils import get_video_bit, calculate_marco_f1, format_conversion, rewrite, load_one_trace, calculate_macro_f1_conf_curve

# VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
VIDEO_BIT_RATE = [200, 800, 1500, 2500, 4000, 7000, 12000, 18000, 32000, 50000, 70000]

# 1080p
# RE = [[720, 480], [1280, 720], [1920, 1080]]
# 720p
# RE = [[426, 240], [854, 480], [1280, 720]]
# DETRAC
# RE = [[426, 240], [640, 360], [960, 540]]
# DSEC
# RE = [[640, 480], [960, 720], [1440, 1080]]
# LMOT
RE = [[864, 480], [1296, 720], [1800, 1000]]

# D²-City_1080p、DETRAC、D²-City_720p
# SKIP = [0, 1, 2, 5]
# LMOT、DSEC
SKIP = [0, 1, 2, 4]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def get_sequence_lengths(video_sequences):
    """
    获取每个视频序列的文件数量，文件数量代表每个序列的长度。

    Arguments:
    - video_sequences_path: 存储视频序列文件夹的路径

    Returns:
    - sequence_lengths: 每个视频序列的长度（即文件数量列表）
    """
    sequence_lengths = []

    # 遍历每个视频序列文件夹，获取文件数量
    for sequence_folder in video_sequences:
        sequence_folder_path = os.path.join(FRAMES_BLUR, sequence_folder)

        if os.path.isdir(sequence_folder_path):
            # 统计当前文件夹下的文件数量
            num_files = len(
                [f for f in os.listdir(sequence_folder_path) if os.path.isfile(os.path.join(sequence_folder_path, f))])
            sequence_lengths.append(num_files)

    return sequence_lengths

def frame_skip(source_folder, target_folder, skip):
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    images = sorted([f for f in os.listdir(source_folder) if f.endswith('.jpg')])
    target_index = 0
    for i in range(0, len(images), skip + 1):
        src_img = images[i]
        dst_img = f'{target_index:02d}.jpg'
        shutil.copy(os.path.join(source_folder, src_img), os.path.join(target_folder, dst_img))
        target_index += 1
    return target_index

def create_temp_dir(bs_path, ls_path, sequence, chunk_no):
    chunk_start = chunk_no * FRAMES_PER_CHUNK
    chunk_end = chunk_start + FRAMES_PER_CHUNK
    print("check start ",chunk_start, chunk_end)

    # 创建临时帧目录
    blur_chunk = f'/mnt/mydisk/lyra/input/{NAME}_{STATUS}_blur/{sequence:03d}_{chunk_no:03d}'
    label_chunk = f'/mnt/mydisk/lyra/input/{NAME}_{STATUS}_truth/{sequence:03d}_{chunk_no:03d}'
    os.makedirs(blur_chunk, exist_ok=True)
    os.makedirs(label_chunk, exist_ok=True)

    for i in range(chunk_start, chunk_end):
        shutil.copy(os.path.join(bs_path, f"{i+1:06d}.png"),
                    os.path.join(blur_chunk, f"{i + 1 - chunk_start:02d}.png"))
        shutil.copy(os.path.join(ls_path, f"{i+1:06d}.txt"),
                    os.path.join(label_chunk, f"{sequence:03d}_{chunk_no:03d}_{i + 1 - chunk_start}.txt"))

def encoder(image_folder, video_name, w, h, fps, skip, bit):
    directory = os.path.dirname(video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = fps / (skip + 1)
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-start_number', '0',
        '-i', os.path.join(image_folder, '%02d.png'),
        '-vf', f"select='not(mod(n,{skip + 1}))',setpts=N/({f}*TB),scale={w}:{h}",
        '-frames:v', str(FRAMES_PER_CHUNK // (skip + 1)),
        '-r', str(f),
        '-c:v', 'libx264',
        '-b:v', f'{bit}k',
        video_name
    ]
    start_time = time.time()
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ct = time.time() - start_time
    return ct

# 检测 → 坐标要处理一下
def detect_data(model, video_name, output_folder, num, frame):
    num = str(num)
    base_detect = Path("/mnt/mydisk/lyra/runs/detect")
    base_input = Path("/mnt/mydisk/lyra/input")
    detect_target = base_detect / output_folder / num
    input_target = base_input / output_folder / num
    # 清理历史输出（如果存在）
    if detect_target.exists():
        shutil.rmtree(detect_target)
    if input_target.exists():
        shutil.rmtree(input_target)
    base_detect.mkdir(parents=True, exist_ok=True)

    src = Path(video_name)
    if src.is_dir():
        # 可选：过滤目录内非图片文件（YOLO 可以处理，但显式更安全）
        imgs = [p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')]
        if not imgs:
            raise ValueError(f"目录 {video_name} 中没有支持的图片文件。")
        yolo_source = str(src)  # 传目录给 YOLO
    elif src.is_file():
        yolo_source = str(src)  # 传 mp4 文件给 YOLO（向后兼容）
    else:
        raise ValueError(f"source_path {video_name} 不存在或不是文件/目录。")
    model(
        source=yolo_source,
        save_txt=True,
        save_conf=True,
        project=str(base_detect),
        name=f'{output_folder}/{num}',
        conf=0.001,
        iou=0.4,
        imgsz=960,
    )
    # 检查 labels 文件夹是否存在；如果不存在，创建一个空文件夹以免 format_conversion 报错
    labels_dir = base_detect / output_folder / num / "labels"
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)

    # 创建目标 input 目录并调用格式转换
    input_target.mkdir(parents=True, exist_ok=True)
    format_conversion(str(labels_dir), str(input_target) + '/', 1800, 1000)
    # 父目录，用于 rewrite 补齐
    parent_directory = str(base_input / output_folder)
    # rewrite 的作用：用空白 txt 补齐缺失的帧标签，保证有 frame 个 txt
    rewrite(parent_directory, num, frame)

def recovery_label(name, num, skip, frame):
    source = f'/mnt/mydisk/lyra/input/{name}/{num}'
    target = f'/mnt/mydisk/lyra/input/{name}/{num}_r'
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target)
    for i in range(1, math.ceil(frame / (skip + 1)) + 1):
        src_file_name = f"{num}_{i}.txt"  # Original file name
        if not os.path.exists(os.path.join(source, src_file_name)):
            src_file_name = f"{num}_{i-1}.txt"
            shutil.copy(os.path.join(source, src_file_name),
                        os.path.join(target, f'{num}_{1 + (skip + 1) * (i - 1)}.txt'))
        else:
            shutil.copy(os.path.join(source, src_file_name), os.path.join(target, f'{num}_{1 + (skip + 1) * (i - 1)}.txt'))
        for j in range(1, skip + 1):
            if 1 + (skip + 1) * (i - 1) + j > frame:
                break
            dst_file_name = f"{num}_{1 + (skip + 1) * (i - 1) + j}.txt"  # New file name
            src_file_path = os.path.join(source, src_file_name)
            dst_file_path = os.path.join(target, dst_file_name)
            shutil.copy(src_file_path, dst_file_path)

if __name__ == '__main__':
    # REDS 数据路径
    FRAMES_BLUR = '/mnt/mydisk/lyra/RL_Dataset/LMOT/images'
    RAW_truth = '/mnt/mydisk/lyra/RL_Dataset/LMOT/sort'
    BEST_WEIGHTS = '/home/ubuntu/lyra/MPC/bifpn_cpa+x/bc.pt'
    NAME = 'LMOT'
    STATUS = 'train'

    FPS = 20
    Length = 2  # seconds per chunk
    FRAMES_PER_CHUNK = int(FPS * Length)  # = 25

    model = YOLO(BEST_WEIGHTS)  # 这里就是 6 类版本的模型
    print("模型类别数:", len(model.names))
    print("类别映射:", model.names)  # 确认一下 0~5 对应什么

    # SEQUENCES = sorted(os.listdir(FRAMES_BLUR))[6:]
    all_seqs = sorted(os.listdir(FRAMES_BLUR))
    # idxs = [1, 4, 13, 27, 29, 36, 40, 44, 46, 64, 81, 83, 88, 98, 102] # D²-City
    # idxs = [7, 11, 14, 15] # DETRAC
    # idxs = [1, 4, 6]  # DSEC
    idxs = [6, 7, 8]  # LMOT
    SEQUENCES = [all_seqs[i] for i in idxs if i < len(all_seqs)]

    # 初始化 HDF5 表格结构
    seq_lengths = get_sequence_lengths(SEQUENCES)
    # print(seq_lengths)
    # 动态计算 CHUNK 数量
    chunks = [length // FRAMES_PER_CHUNK for length in seq_lengths]
    print("check chunks ",chunks)

    # 创建动态的 MultiIndex
    index_list = []

    for seq_idx in range(len(seq_lengths)):
        chunk_count = chunks[seq_idx]  # 获取该序列的 CHUNK 数量
        # 生成该序列的 CHUNK 索引
        index_list.extend([(seq_idx, chunk, bit, re) for chunk in range(chunk_count)
                           for bit in range(11)  for re in range(3)])

    # 创建 MultiIndex
    index = pd.MultiIndex.from_tuples(index_list, names=['SEQ', 'CHUNK', 'BIT', 'RE'])
    metrics = ['Size', 'Accuracy', 'Bitrate']

    # Save to HDF5
    if not os.path.exists('h5_file'):
        os.makedirs('h5_file', exist_ok=True)
    h5_path = f'h5_file/{NAME}_{STATUS}_DAO.h5'
    # ✅ 如果文件已存在则读取，否则新建
    if not os.path.exists(h5_path):
        df = pd.DataFrame(index=index, columns=metrics).fillna(0)
        df.to_hdf(h5_path, key='encoding_data', mode='w')
    # -----------------------------------------------------------
    df = pd.read_hdf(h5_path, 'encoding_data')
    # 编码后视频输出地址
    video_path = f'/mnt/mydisk/lyra/dataset/video_{NAME}_{STATUS}'

    seq_id = 0
    # 生成编码配置的所有组合
    encoding_configs = [(j, n) for j in range(11) for n in range(3)]
    avg_encoding_times = {config: [] for config in encoding_configs}  # 用于存储每个配置的编码时间
    for seq in SEQUENCES:
        # if seq_id <= 50:
        #     seq_id += 1
        #     continue
        # if seq_id >= 100:
        #     break
        blur_seq_path = os.path.join(FRAMES_BLUR, seq)
        label_seq_path = os.path.join(RAW_truth, seq)
        for chunk_id in range(chunks[seq_id]):
            # if seq_id == 51 and chunk_id <= 27:
            #     continue
            create_temp_dir(blur_seq_path, label_seq_path, seq_id, chunk_id)
            for j in range(11):  # BIT
                for n in range(3):  # RE
                    output_video = f'{video_path}/{seq_id:03d}_{chunk_id:03d}.mp4'
                    coding_time = encoder(f'/mnt/mydisk/lyra/input/{NAME}_{STATUS}_blur/{seq_id:03d}_{chunk_id:03d}',
                                          output_video, RE[n][0], RE[n][1], FPS, SKIP[0], VIDEO_BIT_RATE[j])
                    avg_encoding_times[(j, n)].append(coding_time)
                    video_chunk_size = os.path.getsize(output_video)
                    real_bit = get_video_bit(output_video)
                    df.loc[(seq_id, chunk_id, j, n), 'Size'] = video_chunk_size
                    df.loc[(seq_id, chunk_id, j, n), 'Bitrate'] = real_bit

                    detect_data(model, output_video, f'{NAME}_{STATUS}', f'{seq_id:03d}_{chunk_id:03d}',
                                math.floor(FRAMES_PER_CHUNK / (SKIP[0] + 1)))
                    _, f = calculate_macro_f1_conf_curve(f'{NAME}_{STATUS}_truth',
                                                         f'{NAME}_{STATUS}',
                                                         f'{seq_id:03d}_{chunk_id:03d}',
                                                         f'{seq_id:03d}_{chunk_id:03d}', FRAMES_PER_CHUNK)
                    df.loc[(seq_id, chunk_id, j, n), 'Accuracy'] = f
            df.to_hdf(f'h5_file/{NAME}_{STATUS}_DAO.h5', key='encoding_data', mode='w')
        df.to_hdf(f'h5_file/{NAME}_{STATUS}_DAO.h5', key='encoding_data', mode='w')
        seq_id += 1
    df.to_hdf(f'h5_file/{NAME}_{STATUS}_DAO.h5', key='encoding_data', mode='w')

    if STATUS == "train":
        if not os.path.exists("coding_time"):
            os.makedirs("coding_time", exist_ok=True)

        all_avg_encoding_times = {config: 0.0 for config in encoding_configs}

        # 计算每种编码配置的平均编码时间
        for config in encoding_configs:
            all_avg_encoding_times[config] = np.mean(avg_encoding_times[config])

            # 写入结果到CSV文件
        with open(f'coding_time/coding_time_{NAME}.csv', mode='w', newline='') as csvfile:
            fieldnames = ['BIT', 'RE', 'mean_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for config, avg_encoding_times in all_avg_encoding_times.items():
                j, n = config
                writer.writerow({
                    'BIT': j,
                    'RE': n,
                    'mean_time': f"{avg_encoding_times:.16f}"
                })

