import numpy as np
import torch
import os
import shutil
import subprocess
import cv2
from ultralytics import YOLO
from collections import defaultdict

# 单位是Kb/s
VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]

# 1080p
# resolutions = ['480p', '720p', '1080p']
# re = [[720, 480], [1280, 720], [1920, 1080]]
# 720p
# resolutions = ['240p', '480p', '720p']
# re = [[426, 240], [854, 480], [1280, 720]]
# DETRAC
# resolutions = ['240p', '360p', '540p']
# re = [[426, 240], [640, 360], [960, 540]]
# DSEC
resolutions = ['480p', '720p', '1080p']
re = [[640, 480], [960, 720], [1440, 1080]]
# LMOT
# resolutions = ['480p', '720p', '1000p']
# re = [[864, 480], [1296, 720], [1800, 1000]]

NAME = 'DSEC'
button = 33
MODEL_WEIGHTS = '/home/ubuntu/lyra/MPC/sample_cpa.pt'

def load_one_trace(trace_folder, n):
    trace_files = sorted(os.listdir(trace_folder))
    # for item in trace_files:
    #     print(item)
    file_path = trace_folder + trace_files[n]
    s = []
    with open(file_path, 'rb') as f:
        for line in f:
            s.append(float(line))
    return s, trace_files[n]

def C_R(a, l):
    return (2 * a if a >= 0.7 else 0) - l

def bandwidth_predictor(bw, est):
    # if time >= est:
    #     r = sum(bw[time - est:time])/est
    # else:
    #     r = sum(bw[0:time])/time
    # return r
    if est > len(bw):
        r = sum(bw)/len(bw)
    else:
        r = sum(bw[-est:]) / len(bw[-est:])
    return r


def encoder(image_folder, video_name, bit, w, h, l, fps):
    directory = os.path.dirname(video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # input_path = os.path.join(image_folder, '%02d.JPEG')  # 图片命名为 00.JPEG, 01.JPEG, ...
    input_path = os.path.join(image_folder, '%02d.jpg')
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', str(bit) + 'k',
        '-s', f'{w}x{h}',
        '-t', f'{l}',
        video_name
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect_data(video_name, output_folder, num):
    if os.path.isdir(f'runs/detect/{output_folder}/{num}'):
        shutil.rmtree(f'runs/detect/{output_folder}/{num}')
    if os.path.isdir(f'input/{output_folder}/{num}'):
        shutil.rmtree(f'input/{output_folder}/{num}')
    yolo = YOLO('yolov8s.pt', task="detect")
    yolo(source=video_name, save_txt=True, save_conf=True, name=f'{output_folder}/{num}')
    new_p = f'input/{output_folder}/{num}'
    folder = os.path.exists(new_p)
    if not folder:
        os.makedirs(new_p)
    format_conversion(f'runs/detect/{output_folder}/{num}/labels', f'{new_p}/', 3840, 2160)
    parent_directory = f'input/{output_folder}'
    rewrite(parent_directory, num)


def load_trace(trace_folder):
    trace_files = os.listdir(trace_folder)
    all_trace = []
    trace_name = []
    for trace_file in trace_files:
        file_path = trace_folder + trace_file
        s = []
        with open(file_path, 'rb') as f:
            for line in f:
                s.append(float(line))
        all_trace.append(s)
        trace_name.append(trace_file)
    return all_trace, trace_name


def get_video_bit(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
    cap.release()
    return bitrate


def jpeg_video(image_folder, output_folder, bit, w, h, frames, fps):
    if not output_folder:
        os.makedirs(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')])
    # segments = [image_files[i:i + frames_per_segment] for i in range(0, len(image_files), frames_per_segment)]
    segments = [image_files[i:i + frames] for i in range(0, len(image_files), frames) if
                len(image_files[i:i + frames]) == frames]
    n = 0
    for i, segment in enumerate(segments):
        output_filename = os.path.join(output_folder, f'{i:03d}.mp4')
        temp_list_path = f'{NAME}_temp_list.txt'
        with open(temp_list_path, 'w') as file:
            for img in segment:
                file.write(f"file '{os.path.join(image_folder, img)}'\n")
                file.write(f"duration {1 / fps}\n")
            file.write(f"file '{os.path.join(image_folder, segment[-1])}'\n")
        ffmpeg_command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_path,
            '-c:v', 'libx264',
            '-b:v', f'{bit}k',
            '-vf', f'scale={w}:{h}',
            '-r', str(fps),
            '-frames:v', str(frames),
            output_filename
        ]
        subprocess.run(ffmpeg_command)
        # print(f"Segment {i+1} has been processed.")
        os.remove(temp_list_path)
        n = n + 1
    return n


def process_all_folders(raw_data_folder, output_folder, dataset, frames, fps):
    # Iterate over all folders in the raw data directory
    # video_id = 0
    all_seqs = sorted(os.listdir(raw_data_folder))
    # idxs = [1, 4, 13, 27, 29, 36, 40, 44, 46, 64, 81, 83, 88, 98, 102] # D²-City
    # idxs = [7, 11, 14, 15] # DETRAC
    idxs = [1, 4, 6]  # DSEC
    SEQUENCES = [all_seqs[i] for i in idxs if i < len(all_seqs)]
    sample = 0
    f1 = []
    for folder in SEQUENCES:
        # if video_id >= 100:
        #     break
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
        folder_path = os.path.join(raw_data_folder, folder)
        if not output_folder:
            os.makedirs(output_folder)
        if os.path.isdir(folder_path):
            # Create a corresponding output folder for each input folder
            num = 1
            part = 0
            ext = os.path.splitext(os.path.basename(os.listdir(folder_path)[0]))[1]
            for r in re:
                for b in VIDEO_BIT_RATE:
                    output_path = os.path.join(output_folder, f'{num}')
                    if not output_path:
                        os.makedirs(output_path)
                    part = jpeg_video(folder_path, output_path, b, r[0], r[1], frames, fps)
                    num = num + 1
            for p in range(0, part):
                f_p = []
                sample = sample + 1
                flag = p*fps
                # copy_rename_image(folder_path, f"{flag:03d}.jpg", dataset, f"{sample:04d}.jpg")
                copy_rename_image(folder_path, f"{flag:06d}{ext}", dataset, f"{sample:04d}{ext}")
                # base
                train_detect(output_folder, button, p, frames, 1440, 1080)
                base_label(f'input/{NAME}/{button}')
                # other
                for k in range(1, button):
                    train_detect(output_folder, k, p, frames, 1440, 1080)
                    f = train_f1(f'{button}', f'{k}', p, frames)
                    f_p.append(f)
                f_p.append(float(1))
                f1.append(f_p)
        # video_id = video_id + 1
    return f1


def train_detect(video_name, num, p, frames, w, h):
    if os.path.isdir(f'runs/detect/{NAME}/{num}'):
        shutil.rmtree(f'runs/detect/{NAME}/{num}')
    if os.path.isdir(f'input/{NAME}/{num}'):
        shutil.rmtree(f'input/{NAME}/{num}')
    yolo = YOLO(MODEL_WEIGHTS, task="detect")
    # yolo(
    #     source=f'{video_name}/{num}/{p:03d}.mp4',
    #     save_txt=True,
    #     save_conf=True,
    #     name=f'{NAME}/{num}'
    # )
    yolo(source=f'{video_name}/{num}/{p:03d}.mp4', save_txt=True, save_conf=True, name=f'{NAME}/{num}')
    new_p = f'input/{NAME}/{num}'
    folder = os.path.exists(new_p)
    if not folder:
        os.makedirs(new_p)
    format_conversion(f'runs/detect/{NAME}/{num}/labels', f'{new_p}/', w, h)
    parent_directory = f'input/{NAME}'
    rewrite_one(parent_directory, num, p, frames)


def rewrite(v_path, num):
    folder_name = os.path.join(v_path, str(num))
    files = [f for f in os.listdir(folder_name) if f.endswith('.txt')]
    if len(files) < 40:
        for j in range(1, 41):
            file_name = os.path.join(folder_name, f"{num}_{j}.txt")
            if not os.path.exists(file_name):
                with open(file_name, 'w') as fp:
                    pass


def rewrite_one(v_path, num, p, frames):
    folder_name = os.path.join(v_path, str(num))
    files = [f for f in os.listdir(folder_name) if f.endswith('.txt')]
    if len(files) < frames:
        for j in range(1, frames + 1):
            file_name = os.path.join(folder_name, f"{p:03d}_{j}.txt")
            if not os.path.exists(file_name):
                with open(file_name, 'w') as fp:
                    pass


def copy_rename_image(src_folder, src_filename, dst_folder, dst_filename):
    src_file_path = os.path.join(src_folder, src_filename)
    dst_file_path = os.path.join(dst_folder, dst_filename)
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src_file_path, dst_file_path)


def format_conversion(label_path, new_path, width: int, height: int):
    folder_list = os.listdir(label_path)
    for i in folder_list:
        label_path_new = os.path.join(label_path, i)
        with open(label_path_new, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        lb[:, 1:] = vertex_coo(lb[:, 1:], width, height)
        for _, x in enumerate(lb):
            with open(new_path + i, 'a') as fw:
                fw.write(
                    str(int(x[0])) + ' ' + str(x[5]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + ' ' + str(
                        x[4]) + '\n')


def vertex_coo(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def read_gt(file_path):
    """
    Read ground truth data from a file.
    """
    with open(file_path, 'r') as f:
        boxes = [list(map(float, line.split())) for line in f.readlines()]
    return boxes


def read_detections(file_path):
    """
    Read detection truth data from a file.
    """
    with open(file_path, 'r') as f:
        boxes = []
        for line in f.readlines():
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            # For detection files, skip the confidence score which is the second value
            boxes.append([class_id] + parts[2:])
    return boxes


def train_f1(gt_name, det_name, p, frames):
    gt_path = f'input/{NAME}/{gt_name}'
    det_path = f'input/{NAME}/{det_name}'
    gt_files = [os.path.join(gt_path, f"{p:03d}_{i}.txt") for i in range(1, frames+1)]
    det_files = [os.path.join(det_path, f"{p:03d}_{i}.txt") for i in range(1, frames+1)]

    # all_gt_boxes = defaultdict(list)
    # all_pred_boxes = defaultdict(list)
    #
    # for gt_file, det_file in zip(gt_files, det_files):
    #     for box in read_gt(gt_file):
    #         all_gt_boxes[box[0]].append(box[1:])
    #     for box in read_detections(det_file):
    #         all_pred_boxes[box[0]].append(box[1:])
    #
    # f1_scores = []
    # for cls in all_gt_boxes:
    #     gt_boxes = all_gt_boxes[cls]
    #     pred_boxes = all_pred_boxes[cls]
    #     f1 = calculate_f1_for_class(gt_boxes, pred_boxes)
    #     f1_scores.append(f1)

    # marco_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    all_frame_f1 = []

    for gt_file, det_file in zip(gt_files, det_files):
        all_gt_boxes = defaultdict(list)
        all_pred_boxes = defaultdict(list)
        for box in read_gt(gt_file):
            cls = int(box[0])
            all_gt_boxes[cls].append(box[1:])
        for box in read_detections(det_file):
            all_pred_boxes[box[0]].append(box[1:])
        f1_scores = []
        for cls in all_gt_boxes:
            gt_boxes = all_gt_boxes[cls]
            pred_boxes = all_pred_boxes[cls]
            f1 = calculate_f1_for_class(gt_boxes, pred_boxes)
            f1_scores.append(f1)
        single_marco_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        all_frame_f1.append(single_marco_f1)

    marco_f1 = sum(all_frame_f1) / len(all_frame_f1) if all_frame_f1 else 0
    return marco_f1


def calculate_marco_f1(gt_name, det_name, num):
    gt_path = f'input/{gt_name}/{num}'
    det_path = f'input/{det_name}/{num}'
    gt_files = [os.path.join(gt_path, f"{num}_{i}.txt") for i in range(1, 61)]
    det_files = [os.path.join(det_path, f"{num}_{i}.txt") for i in range(1, 61)]

    all_gt_boxes = defaultdict(list)
    all_pred_boxes = defaultdict(list)

    for gt_file, det_file in zip(gt_files, det_files):
        for box in read_gt(gt_file):
            all_gt_boxes[box[0]].append(box[1:])
        for box in read_detections(det_file):
            all_pred_boxes[box[0]].append(box[1:])

    f1_scores = []
    for cls in all_gt_boxes:
        gt_boxes = all_gt_boxes[cls]
        pred_boxes = all_pred_boxes[cls]
        f1 = calculate_f1_for_class(gt_boxes, pred_boxes)
        f1_scores.append(f1)

    marco_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return marco_f1


# def calculate_f1_for_class(gt_boxes, pred_boxes):
#     """
#     Calculate F1 score for a single class.
#     """
#     tp = 0  # True positives
#     fp = 0  # False positives
#     fn = 0  # False negatives
#
#     # Calculate true positives and false negatives
#     for gt in gt_boxes:
#         if any(calculate_iou(gt, pred) >= 0.5 for pred in pred_boxes):
#             tp += 1
#         else:
#             fn += 1
#
#     # Calculate false positives
#     for pred in pred_boxes:
#         if not any(calculate_iou(gt, pred) >= 0.5 for gt in gt_boxes):
#             fp += 1
#
#     # Calculate precision, recall, and F1
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     return f1

def calculate_f1_for_class(gt_boxes, pred_boxes):
    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    tp = 0

    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            iou_val = calculate_iou(pred, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= 0.5 and best_j != -1:
            tp += 1
            matched_gt[best_j] = True
            matched_pred[i] = True

    fp = matched_pred.count(False)
    fn = matched_gt.count(False)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def base_label(folder_path):
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        # 打开文件
        with open(file_path, 'r') as file:
            lines = file.readlines()
        column_to_delete = 1
        for i in range(len(lines)):
            columns = lines[i].split()
            del columns[column_to_delete]
            lines[i] = ' '.join(columns) + '\n'
        with open(file_path, 'w') as file:
            file.writelines(lines)
    print("done")

