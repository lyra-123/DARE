import numpy as np
import pandas as pd
import os
import subprocess
import shutil
from ultralytics import YOLO
from collections import defaultdict
from DAO_utils import process_all_folders

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]

# 1080p
# resolutions = ['480p', '720p', '1080p']
# re = [[720, 480], [1280, 720], [1920, 1080]]
# 720p
# resolutions = ['240p', '480p', '720p']
# RE = [[426, 240], [854, 480], [1280, 720]]
# DETRAC
# resolutions = ['240p', '360p', '540p']
# re = [[426, 240], [640, 360], [960, 540]]
# DSEC
resolutions = ['480p', '720p', '1080p']
re = [[640, 480], [960, 720], [1440, 1080]]
# LMOT
# resolutions = ['480p', '720p', '1000p']
# re = [[864, 480], [1296, 720], [1800, 1000]]

button = 33
Fps = 20
segment_length = 2
frames_per_segment = Fps * segment_length
raw_data = '/mnt/mydisk/lyra/RL_Dataset/DSEC/images'
NAME = 'DSEC'
dataset = f'dataset/{NAME}'
mid_folder = f'mid/{NAME}'


if __name__ == '__main__':
    # 这一步和构建h5文件类似，原始的raw_data找不到了，换成其他数据集。得到的数据形式参考'dataset/train'
    f1_scores = process_all_folders(raw_data, mid_folder, dataset, frames_per_segment, Fps)
    # Constructing the column names for the CSV
    column_names = ['frame_path'] + [f'f1_score_{bitrate}_{res}' for res in resolutions for bitrate in VIDEO_BIT_RATE]

    # Creating the frame_path column
    frame_paths = [f"{i:04d}.jpg" for i in range(1, len(f1_scores)+1)]  # Assuming file names are 001.png, 002.png, ...

    # Combining frame_paths with f1_scores into a single DataFrame
    data = np.column_stack((frame_paths, f1_scores))
    df = pd.DataFrame(data, columns=column_names)

    # Save the DataFrame to a CSV file
    csv_file_path = f'dataset/{NAME}/scores.csv'
    df.to_csv(csv_file_path, index=False)
