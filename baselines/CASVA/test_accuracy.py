import os
import subprocess
import shutil
from ultralytics import YOLO
import math

from utils import format_conversion, rewrite, calculate_marco_f1


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
        '-frames:v', str(40 // (skip + 1)),
        '-r', str(f),
        '-c:v', 'libx264',
        '-x264-params', f'qp={qp}',
        video_name
    ]
    # try:
    #     subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # except subprocess.CalledProcessError as e:
    #     print("FFmpeg 报错了：")
    #     print(e.stderr.decode())
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 检测 → 坐标要处理一下
def detect_data(video_name, output_folder, num, frame):
    if os.path.isdir(f'runs/detect/{output_folder}/{num}'):
        shutil.rmtree(f'runs/detect/{output_folder}/{num}')
    if os.path.isdir(f'input/{output_folder}/{num}'):
        shutil.rmtree(f'input/{output_folder}/{num}')
    yolo = YOLO("yolov8s.pt", task="detect")
    yolo(source=video_name, save_txt=True, save_conf=True, name=f'{output_folder}/{num}', classes=[0, 1, 2, 3, 5, 7])
    new_p = f'input/{output_folder}/{num}'
    os.makedirs(new_p, exist_ok=True)
    format_conversion(f'runs/detect/{output_folder}/{num}/labels', f'{new_p}/', 1440, 1080)
    parent_directory = f'input/{output_folder}'
    # 这一步的作用是防止某些帧没检测出目标，导致没有生成相应的标签文件，此时调用rewarite会用空白文件来补齐frame个txt文件
    rewrite(parent_directory, num, frame)

def create_temp_dir(bs_path, ls_path, sequence, chunk_no):
    chunk_start = chunk_no * 50
    chunk_end = chunk_start + 50
    print("check start ",chunk_start, chunk_end)

    # 创建临时帧目录
    blur_chunk = f'input/{NAME}_{STATUS}_blur/{sequence:03d}_{chunk_no:03d}'
    label_chunk = f'input/{NAME}_{STATUS}_truth/{sequence:03d}_{chunk_no:03d}'
    os.makedirs(blur_chunk, exist_ok=True)
    os.makedirs(label_chunk, exist_ok=True)

    for i in range(chunk_start, chunk_end):
        shutil.copy(os.path.join(bs_path, f"{i:06d}.png"),
                    os.path.join(blur_chunk, f"{i + 1 - chunk_start:02d}.png"))
        shutil.copy(os.path.join(ls_path, f"{i:06d}.txt"),
                    os.path.join(label_chunk, f"{sequence:03d}_{chunk_no:03d}_{i + 1 - chunk_start}.txt"))

FRAMES_BLUR = '/home/dell/lyra/Dataset/DSEC/train/images'
RAW_truth = '/home/dell/lyra/Dataset/DSEC/train/labels'
NAME = 'DSEC'
STATUS = 'train3'
SEQUENCES = sorted(os.listdir(FRAMES_BLUR))
blur_seq_path = os.path.join(FRAMES_BLUR, SEQUENCES[0])
label_seq_path = os.path.join(RAW_truth, SEQUENCES[0])
create_temp_dir(blur_seq_path, label_seq_path, 0, 0)
video_path = f'/home/dell/lyra/CASVA/dataset/video_{NAME}_{STATUS}'
output_video = f'{video_path}/{0:03d}_{0:03d}.mp4'
encoder(f'input/{NAME}_{STATUS}_blur/{0:03d}_{0:03d}', output_video, 1440, 1080, 20, 0, 23)
detect_data(output_video, f'{NAME}_{STATUS}', f'{0:03d}_{0:03d}', math.floor(40 / (0 + 1)))
f = calculate_marco_f1(f'{NAME}_{STATUS}_truth', f'{NAME}_{STATUS}', f'{0:03d}_{0:03d}', f'{0:03d}_{0:03d}', 40)
print("result is ",f)

import os

# def iou(box1, box2):
#     xi1 = max(box1[0], box2[0])
#     yi1 = max(box1[1], box2[1])
#     xi2 = min(box1[2], box2[2])
#     yi2 = min(box1[3], box2[3])
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
#     box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area
#     return inter_area / union_area if union_area > 0 else 0
#
# def load_gt_boxes(path):
#     boxes = []
#     if not os.path.exists(path):
#         return boxes
#     with open(path, 'r') as f:
#         for line in f:
#             parts = list(map(float, line.strip().split()))
#             cls_id = int(parts[0])
#             x1, y1, x2, y2 = parts[1:5]
#             boxes.append([cls_id, x1, y1, x2, y2])
#     return boxes
#
# def load_pred_boxes(path, conf_thresh=0.3):
#     boxes = []
#     if not os.path.exists(path):
#         return boxes
#     with open(path, 'r') as f:
#         for line in f:
#             parts = list(map(float, line.strip().split()))
#             if len(parts) < 6:
#                 continue
#             cls_id = int(parts[0])
#             conf = parts[1]
#             if conf < conf_thresh:
#                 continue
#             x1, y1, x2, y2 = parts[2:6]
#             boxes.append([cls_id, x1, y1, x2, y2])
#     return boxes
#
# def calculate_f1(gt_dir, pred_dir, iou_thresh=0.5, conf_thresh=0.3):
#     tp = fp = fn = 0
#     frame_files = sorted(os.listdir(gt_dir))
#
#     for file in frame_files:
#         idx = file.split('_')[-1].split('.')[0]
#         pred_path = os.path.join(pred_dir, f'output_video_{idx}.txt')
#         gt_boxes = load_gt_boxes(os.path.join(gt_dir, file))
#         pred_boxes = load_pred_boxes(pred_path, conf_thresh=conf_thresh)
#
#         matched = [False] * len(gt_boxes)
#
#         for pred in pred_boxes:
#             pred_cls, pred_box = pred[0], pred[1:]
#             best_iou = 0
#             best_idx = -1
#             for i, gt in enumerate(gt_boxes):
#                 if matched[i] or gt[0] != pred_cls:
#                     continue
#                 iou_val = iou(pred_box, gt[1:])
#                 if iou_val > best_iou:
#                     best_iou = iou_val
#                     best_idx = i
#             if best_iou >= iou_thresh and best_idx != -1:
#                 tp += 1
#                 matched[best_idx] = True
#             else:
#                 fp += 1
#         fn += matched.count(False)
#
#     precision = tp / (tp + fp) if (tp + fp) else 0
#     recall = tp / (tp + fn) if (tp + fn) else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
#
#     print(f"TP={tp}, FP={fp}, FN={fn}")
#     return precision, recall, f1
#
# gt_dir = '/home/dell/lyra/CASVA/input/D²-City_1080p_test_truth/000_000'
# pred_dir = '/home/dell/lyra/CASVA/utils/input/D²-City_test/000_000'
#
# precision, recall, f1 = calculate_f1(gt_dir, pred_dir, iou_thresh=0.5)
# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")