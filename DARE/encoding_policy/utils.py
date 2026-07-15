import os
import shutil
import numpy as np
import cv2
import math
from torchvision.transforms import ToTensor
from collections import defaultdict
from PIL import Image
import torch
from pathlib import Path
import pandas as pd
import concurrent.futures as cf
import h5py


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

def get_seq_chunks_list(video_file, seq_nums):
    chunk_list = sorted(os.listdir(video_file))
    seq_chunks = {seq_id: 0 for seq_id in range(seq_nums)}
    for chunk in chunk_list:
        seq_id = int(chunk.split('_')[0])
        seq_chunks[seq_id] += 1
    return seq_chunks

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

def fast_gae_to_list(rewards, values, gamma=0.95, gae_param=0.97, bootstrap=0.0, standardize=False):
    """
    rewards : list[float]                        # len = T
    values  : list[Tensor] 每个 shape=(1,1)      # len = T
    returns / advantages -> list[Tensor] 同样 (1,1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r = torch.tensor(rewards, device=device, dtype=torch.float32)      # (T,)
    v = torch.cat([v.view(-1) for v in values]).to(device)             # (T,)

    v_next = torch.cat([v[1:], torch.tensor([bootstrap], device=device)])

    deltas = r + gamma * v_next - v                         # δ_t
    T = deltas.size(0)
    adv = torch.zeros(T, device=device)
    A = 0.0
    for t in range(T - 1, -1, -1):
        A = deltas[t] + gamma * gae_param * A
        adv[t] = A

    if standardize:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ret = adv + v
    returns_list    = [ret[i].view(1, 1) for i in range(T)]
    advantages_list = [adv[i].view(1, 1) for i in range(T)]
    return returns_list, advantages_list



def get_chunk_imgs(video_path, chunk_id, frames_per_chunk, model_name):
    image_file = sorted(os.listdir(video_path))
    chunk_start = chunk_id * frames_per_chunk
    chunk_end = chunk_start + frames_per_chunk
    images = []
    for i in range(chunk_start, chunk_end):
        if model_name == "dpen":
            im = cv2.imread(os.path.join(video_path, image_file[i]))
            im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            im = torch.from_numpy(np.float32(im / 255.)).permute(2, 0, 1)
        else:
            im = crop_img(np.array(Image.open(os.path.join(video_path, image_file[i])).convert('RGB')), base=16)
            to_tensor = ToTensor()
            im = to_tensor(im)
        images.append(im)
    return images


def get_de_feature(video_path, chunk_id, frames_per_chunk, de_model, model_name="dbce", ps=64):
    frames = get_chunk_imgs(video_path, chunk_id, frames_per_chunk, model_name)
    chunk_feats = []
    with torch.no_grad():
        for data in frames:
            data = data.unsqueeze(0).cuda()
            patch_feats = []
            if model_name == 'depen':
                _, _, H, W = data.shape
                for h in range((H % ps) // 2, H - ps, ps):
                    for w in range((H % ps) // 2, W - ps, ps):
                        patch = data[:, :, h:h + ps, w:w + ps]
                        feat = de_model(patch)
                        # feat, _, _ = de_model(patch)
                        patch_feats.append(feat.squeeze(0))
                if patch_feats:
                    img_feat = torch.stack(patch_feats).mean(0)  # 128
                    chunk_feats.append(img_feat)
            else:
                fea, inter = de_model(x_query=data, x_key=data)
                chunk_feats.append(fea.squeeze(0))

    if chunk_feats:
        chunk_feat = torch.stack(chunk_feats).mean(0)  # 128
    else:
        if model_name == 'depen':
            chunk_feat = torch.zeros(128)
        else:
            chunk_feat = torch.zeros(256)
    chunk_feat = torch.nn.functional.normalize(chunk_feat, dim=0)
    return chunk_feat.cpu()

def resize_image(image, size):
    ih, iw, _ = image.shape
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
    image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    return image_back

def process_video_sequence(seq_dir, FPS):
    BLOCK_SIZE = FPS * 2
    SELECTED_FRAMES = [0, FPS-1]
    RESIZE_SHAPE = (224, 224)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frame_files = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith(('.jpg', '.png'))])
    num_blocks = len(frame_files) // BLOCK_SIZE
    results = []
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        block_frames = [block_start + idx for idx in SELECTED_FRAMES if block_start + idx < len(frame_files)]
        imgs = []
        for frame_idx in block_frames:
            img_path = os.path.join(seq_dir, frame_files[frame_idx])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Cannot read {img_path}, skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_image(img, RESIZE_SHAPE)
            img = img.astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
        if len(imgs) == len(SELECTED_FRAMES):
            results.append(np.stack(imgs))
    return results

def process_all_videos(base_dir, FPS):
    all_results = []
    for subdir in sorted(os.listdir(base_dir)):
        seq_dir = os.path.join(base_dir, subdir)
        if not os.path.isdir(seq_dir):
            continue
        blocks = process_video_sequence(seq_dir, FPS)
        all_results.append(blocks)
    return all_results

# def get_chunk_data_map(h5_file: str, seq_min=0, seq_max=26, seq_start=0):
#     df = pd.read_hdf(h5_file, key="encoding_data").reset_index()
#     df = df[df['SEQ'].between(seq_min, seq_max)]

#     qps   = sorted(df['QP'].unique())
#     skips = sorted(df['SKIP'].unique())
#     res   = sorted(df['RE'].unique())
#     n_skip, n_re = len(skips), len(res)
#     qp_idx   = {v: i for i,v in enumerate(qps)}
#     skip_idx = {v: i for i,v in enumerate(skips)}
#     re_idx   = {v: i for i,v in enumerate(res)}
#
#     def cfg_id(r):
#         return qp_idx[r['QP']] * (n_skip * n_re) + skip_idx[r['SKIP']] * n_re + re_idx[r['RE']]
#     df['CFG_ID'] = df.apply(cfg_id, axis=1)

#     data_map = {
#         (int(row['SEQ']) + seq_start, int(row['CHUNK']), int(row['CFG_ID'])):
#         (row['Size'], row['Accuracy'], row['Bitrate'])
#         for _, row in df.iterrows()
#     }
#     return data_map

def get_chunk_data_map(df, seq_min=0, seq_max=10, seq_start=0):
    sub_df = df[(df["SEQ"] >= seq_min) & (df["SEQ"] <= seq_max)]

    data_map = {
        (int(row.SEQ - seq_min + seq_start), int(row.CHUNK), int(row.CFG_ID)): (row.Size, row.Accuracy, row.Bitrate)
        for row in sub_df.itertuples(index=False)
    }
    return data_map

def build_cfg_mapping(df):
    qps = sorted(df['QP'].unique())
    skips = sorted(df['SKIP'].unique())
    res = sorted(df['RE'].unique())

    qp_idx = {v: i for i, v in enumerate(qps)}
    skip_idx = {v: i for i, v in enumerate(skips)}
    re_idx = {v: i for i, v in enumerate(res)}

    n_skip, n_re = len(skips), len(res)

    def cfg_id(qp, skip, re):
        return qp_idx[qp] * (n_skip * n_re) + skip_idx[skip] * n_re + re_idx[re]

    def id_to_cfg(cid):
        qp = qps[cid // (n_skip * n_re)]
        rem = cid % (n_skip * n_re)
        skip = skips[rem // n_re]
        re = res[rem % n_re]
        return qp, skip, re

    return cfg_id, id_to_cfg

def load_h5_file(h5_file, seq_min=0, seq_max=26):
    df = pd.read_hdf(h5_file, key="encoding_data").reset_index()
    df = df[df['SEQ'].between(seq_min, seq_max)]

    cfg_id_fn, _ = build_cfg_mapping(df)
    df['CFG_ID'] = df.apply(
        lambda r: cfg_id_fn(r['QP'], r['SKIP'], r['RE']), axis=1
    )

    return df

def get_sorted_config_list(h5_path, seq_min=0, seq_max=25):
    df = pd.read_hdf(h5_path, 'encoding_data').reset_index()
    df = df[df['SEQ'].between(seq_min, seq_max)]
    cfg_id_fn, _ = build_cfg_mapping(df)
    df = df.sort_values(
        by=["SEQ", "CHUNK", "Accuracy", "Size"],
        ascending=[True, True, False, True]
    )
    sorted_config_list = []
    for _, seq_df in df.groupby("SEQ", sort=True):
        seq_config_list = []
        for _, chunk_df in seq_df.groupby("CHUNK"):
            cfg_ids = [
                cfg_id_fn(qp, skip, re)
                for qp, skip, re in chunk_df[["QP", "SKIP", "RE"]].values
            ]
            seq_config_list.append(cfg_ids)
        sorted_config_list.append(seq_config_list)
    return sorted_config_list

def load_deg_h5_as_chunk_map(
    h5_path,
    seq_train_start,
    seq_val_start,
    train_num
):
    deg_map_train = {}
    deg_map_val = {}
    train_seq_chunks = []
    val_seq_chunks = []

    with h5py.File(h5_path, 'r') as f:
        video_ids = sorted(f.keys(), key=lambda x: int(x))
        for local_vid, vid_str in enumerate(video_ids):
            feats = f[vid_str]['features'][:]
            num_chunks = feats.shape[0]
            if local_vid < train_num:
                seq_id = seq_train_start + local_vid
                train_seq_chunks.append(num_chunks)
                for chunk_id in range(num_chunks):
                    deg_map_train[(seq_id, chunk_id)] = feats[chunk_id]
            else:
                seq_id = seq_val_start + (local_vid - train_num)
                val_seq_chunks.append(num_chunks)
                for chunk_id in range(num_chunks):
                    deg_map_val[(seq_id, chunk_id)] = feats[chunk_id]

    return deg_map_train, deg_map_val, train_seq_chunks, val_seq_chunks

