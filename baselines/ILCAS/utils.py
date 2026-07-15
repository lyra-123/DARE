import os
import shutil
import numpy as np
import cv2
import math
from collections import defaultdict
import torch
import torch.nn as nn
import pandas as pd

def C_R(a, l):
    return (2 * a if a >= 0.7 else 0) - l

def load_trace(trace_folder):
    trace_files = sorted(os.listdir(trace_folder))
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


def create_sequential_files(src_dir, dst_dir, segments, frames_per_segment):
    os.makedirs(dst_dir, exist_ok=True)  # Create the destination directory if it doesn't exist

    file_count = 1  # Start naming files with 1.txt
    for segment in range(1, segments + 1):
        for frame in range(1, frames_per_segment + 1):
            src_file_name = f'{segment}_{frame}.txt'
            dst_file_name = f'{file_count}.txt'
            src_file_path = os.path.join(src_dir, str(segment), src_file_name)
            dst_file_path = os.path.join(dst_dir, dst_file_name)

            if os.path.exists(src_file_path):  # Check if the source file exists
                shutil.copy(src_file_path, dst_file_path)  # Copy the file
            else:
                print(f"Missing source file: {src_file_path}")

            file_count += 1  # Increment the file count for naming


def create_truth(raw_truth_path, test_path, truth_path):
    # Ensure the base directory for truth exists
    os.makedirs(truth_path, exist_ok=True)

    # Prepare to iterate over the raw_truth files
    frame_files = sorted(os.listdir(raw_truth_path), key=lambda x: int(x.split('.')[0]))
    frame_iter = iter(frame_files)

    # Walk through the test directory to understand its structure
    for segment in sorted(os.listdir(test_path), key=lambda x: int(x)):
        test_segment_path = os.path.join(test_path, segment)
        truth_segment_path = os.path.join(truth_path, segment)

        # Ensure the truth segment directory exists
        os.makedirs(truth_segment_path, exist_ok=True)

        # Get the number of frames in this test segment
        num_frames = len([name for name in os.listdir(test_segment_path) if name.endswith('.txt')])

        # Copy the frames from raw_truth to the corresponding segment in truth
        for i in range(1, num_frames + 1):
            frame_file = next(frame_iter, None)
            if frame_file is None:
                raise ValueError("Not enough frames in raw_truth to populate truth segments")

            # Rename file to match test segment structure (e.g., "1_1.txt")
            new_frame_name = f"{segment}_{i}.txt"
            shutil.copy(os.path.join(raw_truth_path, frame_file),
                        os.path.join(truth_segment_path, new_frame_name))


def get_files(directory, file_prefix, file_extension=".txt"):
    # Get all files in the directory that match the prefix and extension
    files = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith(file_extension)]
    # Sort the files based on the numerical value after the prefix
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    # Generate the full paths for the files
    full_paths = [os.path.join(directory, f) for f in sorted_files]
    return full_paths


def calculate_marco_f1(gt_name, det_name, num, num_r, frame):

    gt_path = f'input/{gt_name}/{num}'
    det_path = f'input/{det_name}/{num_r}'
    gt_files = [os.path.join(gt_path, f"{num}_{i}.txt") for i in range(1, frame+1)]
    det_files = [os.path.join(det_path, f"{num}_{i}.txt") for i in range(1, frame+1)]
    # gt_files = get_files(gt_path, f"{num}_")
    # det_files = get_files(det_path, f"{num}_")

    # all_gt_boxes = defaultdict(list)
    # all_pred_boxes = defaultdict(list)
    #
    # for gt_file, det_file in zip(gt_files, det_files):
    #     for box in read_gt(gt_file):
    #         cls = int(box[0])
    #         all_gt_boxes[cls].append(box[1:])
    #     for box in read_detections(det_file):
    #         all_pred_boxes[box[0]].append(box[1:])
    #
    # f1_scores = []
    # for cls in all_gt_boxes:
    #     gt_boxes = all_gt_boxes[cls]
    #     pred_boxes = all_pred_boxes[cls]
    #     f1 = calculate_f1_for_class(gt_boxes, pred_boxes)
    #     f1_scores.append(f1)

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


# def calculate_f1_for_class(gt_boxes, pred_boxes):
#     """
#     Calculate F1 score for a single class.
#     """
#     tp = 0  # True positives,正确检测到的框（即检测框与真实框的交并比 IoU 大于等于 0.5）。
#     fp = 0  # False positives,错误检测的框（即检测框没有与任何真实框匹配）。
#     fn = 0  # False negatives,漏检的框（即真实框没有被任何检测框正确匹配）。
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


def format_conversion(label_path, new_path, width: int, height: int):
    folder_list = os.listdir(label_path)
    for i in folder_list:
        label_path_new = os.path.join(label_path, i)
        with open(label_path_new, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        lb[:, 1:] = vertex_coo(lb[:, 1:], width, height)
        for _, x in enumerate(lb):
            # print("check x value: ",x)
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


def rewrite(v_path, num, frame):
    folder_name = os.path.join(v_path, str(num))
    files = [f for f in os.listdir(folder_name) if f.endswith('.txt')]
    if len(files) < frame:
        for j in range(1, frame+1):
            file_name = os.path.join(folder_name, f"{num}_{j}.txt")
            if not os.path.exists(file_name):
                with open(file_name, 'w') as fp:
                    pass

def get_video_bit(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
    cap.release()
    return bitrate


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


def calculate_normalized_areas(boxes, image_width=3840, image_height=2160):
    """ Calculate normalized areas of bounding boxes based on the image dimensions. """
    boxes = np.array(boxes)
    # Check if 'boxes' is empty
    # if boxes.size == 0:
    #     return np.array([])
    if boxes.ndim == 1:
        boxes = np.reshape(boxes, (1, -1))
    widths = (boxes[:, 2] - boxes[:, 0]) / image_width
    heights = (boxes[:, 3] - boxes[:, 1]) / image_height
    areas = widths * heights
    return areas


# def load_bounding_boxes(file_path):
#     """ Load bounding box data from a given text file. """
#     data = np.loadtxt(file_path, delimiter=' ')
#     if data.ndim == 1:
#         data = data.reshape(1, -1)  # Ensure data
#     return data[:, 2:6]  # Assuming columns 2 to 5 are the bounding box coordinates
#
#
# def calculate_centroids(boxes):
#     """ Calculate centroids from bounding box coordinates. """
#     x_center = (boxes[:, 0] + boxes[:, 2]) / 2
#     y_center = (boxes[:, 1] + boxes[:, 3]) / 2
#     return np.stack((x_center, y_center), axis=-1)


# #  Hungarian algorithm (Kuhn-Munkres algorithm)
# def match_boxes(boxes1, boxes2):
#     """ Match boxes based on the Hungarian algorithm using the smallest Euclidean distance of centroids. """
#     centroids1 = calculate_centroids(boxes1)
#     centroids2 = calculate_centroids(boxes2)
#     distances = np.linalg.norm(centroids1[:, None, :] - centroids2[None, :, :], axis=-1)
#
#     # Use the Hungarian algorithm to find the minimum cost matching
#     # This returns row indices and corresponding column indices giving the minimal distance
#     row_ind, col_ind = linear_sum_assignment(distances)
#     return row_ind, col_ind
#
#
# def compute_offsets(boxes_s, boxes_e):
#     """ Compute offsets for matched boxes between two frames using the Hungarian algorithm. """
#     row_ind, col_ind = match_boxes(boxes_s, boxes_e)
#     matched_boxes_frame1 = boxes_s[row_ind]
#     matched_boxes_frame20 = boxes_e[col_ind]
#
#     if len(matched_boxes_frame1) == 0 or len(matched_boxes_frame20) == 0:
#         return np.array([0, 0])
#
#     offsets = np.abs(matched_boxes_frame1[:, :2] - matched_boxes_frame20[:, :2])
#     return np.mean(offsets, axis=0)

def load_boxes(filename):
    """ Load bounding box data from a file including class labels. """
    # # Assuming the class label is in the first column (index 0)
    # return np.loadtxt(filename, usecols=[0, 2, 3, 4, 5])  # Includes class labels now
    data = np.loadtxt(filename, usecols=[0, 2, 3, 4, 5], ndmin=2)
    return data


def cal_iou(box1, box2):
    """ Calculate intersection over union for two boxes. """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def match_boxes(boxes1, boxes2):
    """
    Match boxes between two frames using IoU and class labels.
    Ensure each box in boxes2 matches with only one box in boxes1 and both share the same class.
    """
    matches = []
    used_boxes2 = set()  # To track which boxes in boxes2 are already matched

    for i, box1 in enumerate(boxes1):
        best_iou = 0
        best_j = -1
        for j, box2 in enumerate(boxes2):
            if box2[0] == box1[0] and j not in used_boxes2:  # Check if same class and box2 not already matched
                iou = cal_iou(box1[1:], box2[1:])  # Pass only coordinates to cal_iou
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
        if best_j != -1:
            matches.append((i, best_j))
            used_boxes2.add(best_j)  # Mark this box as matched
        if len(used_boxes2) == len(boxes2):
            break  # Exit the loop early as no more boxes are available for matching
    return matches


def calculate_offsets(boxes1, boxes2):
    """ Calculate offsets for matched boxes. """
    offsets = []
    matches = match_boxes(boxes1, boxes2)
    for i, j in matches:
        center_x1 = (boxes1[i][0] + boxes1[i][2]) / 2
        center_y1 = (boxes1[i][1] + boxes1[i][3]) / 2
        center_x2 = (boxes2[j][0] + boxes2[j][2]) / 2
        center_y2 = (boxes2[j][1] + boxes2[j][3]) / 2
        # Calculate offsets in x and y directions based on center points
        offset_x = center_x2 - center_x1
        offset_y = center_y2 - center_y1
        m = math.sqrt(offset_x ** 2 + offset_y ** 2)/2
        offsets.append(m)
    if len(offsets) == 0:
        mean_offset = 0
    else:
        mean_offset = sum(offsets)/len(offsets)
    return mean_offset

def get_seq_chunks_list_by_h5(h5_file):
    """读取 h5，统计每个 SEQ 的 CHUNK 数量并返回列表"""
    df = pd.read_hdf(h5_file, key="encoding_data")

    # ——— 统计 ———
    if isinstance(df.index, pd.MultiIndex):
        seqs = df.index.get_level_values("SEQ")
        chunks = df.index.get_level_values("CHUNK")
        tmp = pd.DataFrame({"SEQ": seqs, "CHUNK": chunks})
        counts = tmp.groupby("SEQ")["CHUNK"].nunique()
    else:
        if {"SEQ", "CHUNK"} - set(df.columns):
            raise ValueError("数据中缺少 'SEQ' 或 'CHUNK' 列")
        counts = df.groupby("SEQ")["CHUNK"].nunique()

    # ——— 转成列表：位置 == SEQ ———
    max_seq = counts.index.max()               # 列表长度
    chunk_counts = [0] * (max_seq + 1)         # 先全 0
    for seq, num in counts.items():
        chunk_counts[seq] = int(num)

    return chunk_counts

def get_chunk_data_map(df, seq_min=0, seq_max=10, seq_start=0):
    # 1. 先筛选 SEQ 范围
    sub_df = df[(df["SEQ"] >= seq_min) & (df["SEQ"] <= seq_max)]

    # 2. 构造 data_map，并重编号
    data_map = {
        (int(row.SEQ - seq_min + seq_start), int(row.CHUNK), int(row.CFG_ID)): (row.Size, row.Accuracy, row.Bitrate)
        for row in sub_df.itertuples(index=False)
    }
    return data_map

def load_h5_file(h5_file, seq_min=0, seq_max=26):
    df = pd.read_hdf(h5_file, key="encoding_data").reset_index()
    df = df[df['SEQ'].between(seq_min, seq_max)]

    # === 构造唯一 CFG_ID ===
    qps = sorted(df['QP'].unique())
    skips = sorted(df['SKIP'].unique())
    res = sorted(df['RE'].unique())
    n_skip, n_re = len(skips), len(res)
    qp_idx = {v: i for i, v in enumerate(qps)}
    skip_idx = {v: i for i, v in enumerate(skips)}
    re_idx = {v: i for i, v in enumerate(res)}

    def cfg_id(r):
        return qp_idx[r['QP']] * (n_skip * n_re) + skip_idx[r['SKIP']] * n_re + re_idx[r['RE']]

    df['CFG_ID'] = df.apply(cfg_id, axis=1)

    return df