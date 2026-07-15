import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import math
import time
import csv

from ultralytics import YOLO
import cv2
from utils import get_video_bit, calculate_marco_f1, format_conversion, rewrite, load_one_trace

# D²-City_1080p、DETRAC、D²-City_720p
QP = [23, 28, 33, 38, 43]
#DSEC、LMOT
# QP = [18, 23, 28, 33, 38]
# ILCAS
# QP = [21, 25, 29, 33, 37, 41]

# DETRAC
# RE = [[960, 540], [854, 480], [640, 360], [426, 240]]
# D²-City_1080p
# RE = [[1920, 1080], [1280, 720], [960, 540], [720, 480], [320, 240]]
# LMOT
# RE = [[1800, 1000], [1296, 720], [1080, 600], [864, 480]]
# D²-City_720p
# RE = [[1280, 720], [960, 540], [854, 480], [426, 240]]
# DSEC
# RE = [[1440, 1080], [1080, 810], [960, 720], [720, 540], [480, 360]]

# DETRAC
# RE = [[960, 540], [768, 432], [576, 324], [480, 270], [384, 216]]
# LMOT
RE = [[1800, 1000], [1440, 800], [1080, 600], [900, 500], [720, 400]]
# D²-City_720p
# RE = [[1280, 720], [1024, 576], [768, 432], [640, 360], [512, 288]]
# DSEC
# RE = [[1440, 1080], [1152, 864], [864, 648], [720, 540], [576, 432]]

# SKIP对应FPS，FPS = 30/(1+SKIP)
# D²-City_1080p、DETRAC、D²-City_720p
# SKIP = [0, 1, 2, 5]
# LMOT、DSEC
SKIP = [0, 1, 2, 4]
# ILCAS
# SKIP = [0, 1, 2, 5, 11]
# ILCAS
# SKIP = [0, 1, 2, 4, 9]

# CHUNK = 2  # 每个序列只能构成两个chunk
FPS = 20
Length = 2  # seconds per chunk
FRAMES_PER_CHUNK = int(FPS * Length)  # = 50

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# REDS 数据路径
FRAMES_BLUR = '/home/dell/lyra/Dataset/LMOT_dark_rgb_trainval/train/images'
RAW_truth = '/home/dell/lyra/Dataset/LMOT_dark_rgb_trainval/train/labels'
NAME = 'LMOT2'
STATUS = 'train'


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
    blur_chunk = f'input/{NAME}_{STATUS}_blur/{sequence:03d}_{chunk_no:03d}'
    label_chunk = f'input/{NAME}_{STATUS}_truth/{sequence:03d}_{chunk_no:03d}'
    os.makedirs(blur_chunk, exist_ok=True)
    os.makedirs(label_chunk, exist_ok=True)

    for i in range(chunk_start, chunk_end):
        shutil.copy(os.path.join(bs_path, f"{i+1:06d}.png"),
                    os.path.join(blur_chunk, f"{i + 1 - chunk_start:02d}.png"))
        shutil.copy(os.path.join(ls_path, f"{i+1:06d}.txt"),
                    os.path.join(label_chunk, f"{sequence:03d}_{chunk_no:03d}_{i + 1 - chunk_start}.txt"))

def encoder(image_folder, video_name, w, h, fps, skip, qp):
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
        '-x264-params', f'qp={qp}',
        video_name
    ]
    start_time = time.time()
    print("=" * 80)
    print(f"正在编码: {video_name}")
    print(f"输入目录: {image_folder}")
    print(f"配置: QP={qp}, SKIP={skip}, RE={w}x{h}")
    print(f"FFmpeg命令:\n{' '.join(ffmpeg_command)}")
    print("=" * 80)
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ct = time.time() - start_time
    return ct

# 检测 → 坐标要处理一下
def detect_data(video_name, output_folder, num, frame):
    if os.path.isdir(f'runs/detect/{output_folder}/{num}'):
        shutil.rmtree(f'runs/detect/{output_folder}/{num}')
    if os.path.isdir(f'input/{output_folder}/{num}'):
        shutil.rmtree(f'input/{output_folder}/{num}')
    yolo = YOLO("yolov8s.pt", task="detect")
    yolo(source=video_name, save_txt=True, save_conf=True, name=f'{output_folder}/{num}')
    new_p = f'input/{output_folder}/{num}'
    os.makedirs(new_p, exist_ok=True)
    format_conversion(f'runs/detect/{output_folder}/{num}/labels', f'{new_p}/', 1800, 1000)
    parent_directory = f'input/{output_folder}'
    # 这一步的作用是防止某些帧没检测出目标，导致没有生成相应的标签文件，此时调用rewarite会用空白文件来补齐frame个txt文件
    rewrite(parent_directory, num, frame)

def recovery_label(name, num, skip, frame):
    source = f'input/{name}/{num}'
    target = f'input/{name}/{num}_r'
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
    SEQUENCES = sorted(os.listdir(FRAMES_BLUR))
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
        index_list.extend([(seq_idx, chunk, qp, skip, re) for chunk in range(chunk_count)
                           for qp in range(5) for skip in range(4) for re in range(5)])

    # 创建 MultiIndex
    index = pd.MultiIndex.from_tuples(index_list, names=['SEQ', 'CHUNK', 'QP', 'SKIP', 'RE'])
    metrics = ['Size', 'Accuracy', 'Bitrate']
    df = pd.DataFrame(index=index, columns=metrics).fillna(0)

    # Save to HDF5
    if not os.path.exists('h5_file'):
        os.makedirs('h5_file', exist_ok=True)
    df.to_hdf(f'/home/dell/lyra/CASVA/h5_file/{NAME}_{STATUS}.h5', key='encoding_data', mode='w')
    # -----------------------------------------------------------
    df = pd.read_hdf(f'/home/dell/lyra/CASVA/h5_file/{NAME}_{STATUS}.h5', 'encoding_data')
    # 编码后视频输出地址
    video_path = f'dataset/video_{NAME}_{STATUS}'

    seq_id = 0
    # 生成编码配置的所有组合
    encoding_configs = [(j, m, n) for j in range(5) for m in range(4) for n in range(5)]
    avg_encoding_times = {config: [] for config in encoding_configs}  # 用于存储每个配置的编码时间
    for seq in SEQUENCES:
        if seq_id >= 100:
            break
        # if seq_id <= 235:
        #     seq_id += 1
        #     continue
        # if seq_id >= 60:
        #     break
        blur_seq_path = os.path.join(FRAMES_BLUR, seq)
        label_seq_path = os.path.join(RAW_truth, seq)
        for chunk_id in range(chunks[seq_id]):
            if seq_id < 8:
                continue
            # if seq_id == 236 and chunk_id <= 12:
            #     chunk_id += 1
            #     continue
            create_temp_dir(blur_seq_path, label_seq_path, seq_id, chunk_id)
            for j in range(5):  # QP
                for m in range(4):  # SKIP
                    for n in range(5):  # RE
                        output_video = f'{video_path}/{seq_id:03d}_{chunk_id:03d}.mp4'
                        coding_time = encoder(f'input/{NAME}_{STATUS}_blur/{seq_id:03d}_{chunk_id:03d}', output_video, RE[n][0], RE[n][1], FPS, SKIP[m],QP[j])
                        avg_encoding_times[(j, m, n)].append(coding_time)
                        video_chunk_size = os.path.getsize(output_video)
                        real_bit = get_video_bit(output_video)
                        df.loc[(seq_id, chunk_id, j, m, n), 'Size'] = video_chunk_size
                        df.loc[(seq_id, chunk_id, j, m, n), 'Bitrate'] = real_bit

                        detect_data(output_video, f'{NAME}_{STATUS}', f'{seq_id:03d}_{chunk_id:03d}', math.floor(FRAMES_PER_CHUNK / (SKIP[m] + 1)))
                        if m == 0:
                            f = calculate_marco_f1(f'{NAME}_{STATUS}_truth', f'{NAME}_{STATUS}', f'{seq_id:03d}_{chunk_id:03d}', f'{seq_id:03d}_{chunk_id:03d}', FRAMES_PER_CHUNK)
                        else:
                            recovery_label(f'{NAME}_{STATUS}', f'{seq_id:03d}_{chunk_id:03d}', SKIP[m], FRAMES_PER_CHUNK)
                            f = calculate_marco_f1(f'{NAME}_{STATUS}_truth', f'{NAME}_{STATUS}', f'{seq_id:03d}_{chunk_id:03d}', f'{seq_id:03d}_{chunk_id:03d}_r', FRAMES_PER_CHUNK)
                        df.loc[(seq_id, chunk_id, j, m, n), 'Accuracy'] = f
                        break
                    break
                break

            df.to_hdf(f'/home/dell/lyra/CASVA/h5_file/{NAME}_{STATUS}.h5', key='encoding_data', mode='w')
        df.to_hdf(f'/home/dell/lyra/CASVA/h5_file/{NAME}_{STATUS}.h5', key='encoding_data', mode='w')
        seq_id += 1
    df.to_hdf(f'/home/dell/lyra/CASVA/h5_file/{NAME}_{STATUS}.h5', key='encoding_data', mode='w')

    if STATUS == "train":
        if not os.path.exists(f"/home/dell/lyra/CASVA/coding_time"):
            os.makedirs(f"/home/dell/lyra/CASVA/coding_time", exist_ok=True)

        all_avg_encoding_times = {config: 0.0 for config in encoding_configs}

        # 计算每种编码配置的平均编码时间
        for config in encoding_configs:
            all_avg_encoding_times[config] = np.mean(avg_encoding_times[config])

            # 写入结果到CSV文件
        with open(f'/home/dell/lyra/CASVA/coding_time/coding_time_{NAME}.csv', mode='w', newline='') as csvfile:
            fieldnames = ['QP', 'SKIP', 'RE', 'mean_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for config, avg_encoding_times in all_avg_encoding_times.items():
                j, m, n = config
                writer.writerow({
                    'QP': j,
                    'SKIP': m,
                    'RE': n,
                    'mean_time': f"{avg_encoding_times:.16f}"
                })

