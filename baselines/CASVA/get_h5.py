import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import math
from ultralytics import YOLO
import cv2
from utils import get_video_bit, calculate_marco_f1, format_conversion, rewrite

QP = [23, 28, 33, 38, 43]
RE = [[1920, 1080], [1280, 720], [720, 480], [320, 240]]
# SKIP对应FPS，FPS = 30/(1+SKIP)
SKIP = [0, 1, 2, 5]
CHUNK = 1800

FPS = 24
Length = 2


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 原始的视频帧
FRAMES = '../DAO/dataset/Driving3'
# ground truth
RAW_truth = '../Batch_Adaptation/input/Driving3_raw_truth'
NAME = 'Driving3'

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


def encoder(image_folder, video_name, w, h, fps, skip, qp):
    f = fps / (skip + 1)
    directory = os.path.dirname(video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    start = 0
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-start_number', str(start),
        '-i', os.path.join(image_folder, '%02d.jpg'),
        '-vf', f"select='not(mod(n\,{skip + 1}))',setpts=N/({f}*TB),scale={w}:{h}",
        '-frames:v', str(60 // (skip + 1)),
        '-r', str(f),
        '-c:v', 'libx264',  # Ensure you're using the H.264 codec
        '-x264-params', f'qp={qp}',  # Set the QP value
        video_name
    ]
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# 检测 → 坐标要处理一下
def detect_data(video_name, output_folder, num, frame):
    if os.path.isdir(f'runs/detect/{output_folder}/{num}'):
        shutil.rmtree(f'runs/detect/{output_folder}/{num}')
    if os.path.isdir(f'input/{output_folder}/{num}'):
        shutil.rmtree(f'input/{output_folder}/{num}')
    yolo = YOLO("yolov8s.pt", task="detect")
    yolo(source=video_name, save_txt=True, save_conf=True, name=f'{output_folder}/{num}')
    new_p = f'input/{output_folder}/{num}'
    folder = os.path.exists(new_p)
    if not folder:
        os.makedirs(new_p)
    format_conversion(f'runs/detect/{output_folder}/{num}/labels', f'{new_p}/', 3840, 2160)
    parent_directory = f'input/{output_folder}'
    rewrite(parent_directory, num, frame)

# 得到第num个块每帧的检测结果txt文件
def create_truth(start, frames, num):
    source = RAW_truth
    output = f'input/{NAME}_truth/{num}'
    os.makedirs(output, exist_ok=True)
    count = 1  # Initialize the counter for new file names
    for i in range(start, start + frames):
        src_file_name = f"{i}.txt"  # Original file name
        dst_file_name = f"{num}_{count}.txt"  # New file name
        src_file_path = os.path.join(source, src_file_name)
        dst_file_path = os.path.join(output, dst_file_name)
        shutil.copy(src_file_path, dst_file_path)
        count += 1


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
    # 初始化一个空表单
    index = pd.MultiIndex.from_product([range(1800), range(5), range(4), range(4)],
                                       names=['CHUNK', 'QP', 'SKIP', 'RE'])

    metrics = ['Size', 'Accuracy', 'Bitrate']
    df = pd.DataFrame(index=index, columns=metrics).fillna(0)
    # Save to HDF5
    os.makedirs('h5_file', exist_ok=True)
    df.to_hdf(f'h5_file/{NAME}_QP.h5', key='encoding_data', mode='w')
    # -----------------------------------------------------------
    df = pd.read_hdf(f'h5_file/{NAME}_QP.h5', 'encoding_data')
    # 编码后视频输出地址
    video_path = f'dataset/video_{NAME}'

    for i in range(1800):
        index = i + 1
        start = i * 60
        create_truth(start + 1, 60, index)
        # QP:j
        for j in range(5):
            # skip:m
            for m in range(4):
                # re:n
                for n in range(4):
                    encoder(f'{FRAMES}/{index}', f'{video_path}/{index}.mp4', RE[n][0], RE[n][1], FPS, SKIP[m], QP[j])
                    video_chunk_size = os.path.getsize(f'{video_path}/{index}.mp4')
                    real_bit = get_video_bit(f'{video_path}/{index}.mp4')
                    df.loc[(i, j, m, n), 'Bitrate'] = real_bit
                    df.loc[(i, j, m, n), 'Size'] = video_chunk_size

                    detect_data(f'{video_path}/{index}.mp4', NAME, index, math.floor(60 / (SKIP[m] + 1)))
                    if m == 0:
                        f = calculate_marco_f1(f'{NAME}_truth', NAME, index, index, 60)
                    else:
                        recovery_label(NAME, index, SKIP[m], 60)
                        f = calculate_marco_f1(f'{NAME}_truth', NAME, index, f'{index}_r', 60)
                    df.loc[(i, j, m, n), 'Accuracy'] = f
        df.to_hdf(f'{NAME}_QP.h5', key='encoding_data', mode='w')
    df.to_hdf(f'{NAME}_QP.h5', key='encoding_data', mode='w')

