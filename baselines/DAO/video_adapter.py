import numpy as np
from DAO_utils import load_one_trace
from AE.AE_predict import predict
from bisect import bisect_left
import DAO_env
import os
from utils import get_chunk_data_map, get_seq_chunks_list_by_h5, load_h5_file
import pandas as pd
import torch
import time
import resource
import platform

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
resolutions = ['480p', '720p', '1080p']
button = 33

# 1080p
# RE = [[720, 480], [1280, 720], [1920, 1080]]
# 720p
# RE = [[426, 240], [854, 480], [1280, 720]]
# DETRAC
# RE = [[426, 240], [640, 360], [960, 540]]
# DSEC
# RE = [[640, 480], [960, 720], [1440, 1080]]
# LMOT
# RE = [[864, 480], [1296, 720], [1800, 1000]]

RES = [[[426, 240], [640, 360], [960, 540]],
       [[640, 480], [960, 720], [1440, 1080]],
       [[864, 480], [1296, 720], [1800, 1000]],
       [[426, 240], [854, 480], [1280, 720]]]


# NAME = ('DETRAC', 'LMOT', 'D²-City_1080p', 'D²-City_720p', 'DSEC')
# TRACE_FOLDER = '/home/dell/lyra/CASVA/test_trace/4G'
h5_files = ['h5_file/sample_cpa/DETRAC2_train_DAO.h5',
            'h5_file/sample_cpa/DSEC2_train_DAO.h5',
            'h5_file/sample_cpa/LMOT2_train_DAO.h5',
            'h5_file/sample_cpa/D²-City2_train_DAO.h5']
encoding_files = ['coding_time/sample_cpa/coding_time_DETRAC2.csv',
                  'coding_time/sample_cpa/coding_time_DSEC2.csv',
                  'coding_time/sample_cpa/coding_time_LMOT2.csv',
                  'coding_time/sample_cpa/coding_time_D²-City2.csv']
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
# AE模型视频帧的路径
# image = 'dataset/DETRAC/images'
images = ['dataset/sample_cpa/DETRAC/images',
          'dataset/sample_cpa/DSEC/images',
          'dataset/sample_cpa/LMOT/images',
          'dataset/sample_cpa/D²-City/images']
images_xx = ['jpg', 'png', 'png', 'jpg']


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

def test(net_env, video_encoding_time, image, chunk_start, xx, RE, current_video_id, seq_id):
    # # ===============================
    # # 1. 内存 before
    # # ===============================
    # mem_before = get_current_memory_mb()
    # peak_before = get_peak_memory_mb()
    # latencies = []

    bit = 0
    re = 0
    while True:
        # t0 = time.perf_counter()
        et = video_encoding_time[bit * 3 + re]
        r, n, end_of_video = net_env.get_video_chunk(bit, re, et)
        # with open(f'Results/log_test_3.txt', 'a') as f:
        #     f.write(
        #         f'{current_video_id}    {seq_id}    {net_env.video_chunk_counter - 1}    {net_env.F1[-1]}    {net_env.transfer_lag[-1]}   {net_env.Reward[-1]}    {net_env.bw_use[-1]}\n')
        if not end_of_video:
            index = bisect_left(VIDEO_BIT_RATE, r)
            select_bitrate = max(0, index - 1)
            # 输入是下一个视频段的第一帧图像
            # print(f'{image}/{chunk_start+n:04d}.{xx}')
            pre_score = predict(f'{image}/{chunk_start+n:04d}.{xx}')
            pre_score = pre_score[0]
            button_list = np.zeros((len(VIDEO_BIT_RATE), len(RE)))
            convert(pre_score, button_list, len(VIDEO_BIT_RATE), len(RE))
            sub_array = button_list[:select_bitrate, :]
            if sub_array.size == 0:
                bit = 0
                re = 0
            else:
                max_pos = np.unravel_index(np.argmax(sub_array), sub_array.shape)
                bit = max_pos[0]
                re = max_pos[1]
            # t1 = time.perf_counter()
            # latencies.append((t1 - t0) * 1000)
        else:
            # # ===============================
            # # 6. 输出统计结果
            # # ===============================
            # latencies = torch.tensor(latencies)
            # print("\n【推理时间】")
            # print(f"  mean: {latencies.mean():.6f} ms")
            # print(f"  max:  {latencies.max():.6f} ms")
            # print(f"  min:  {latencies.min():.6f} ms")
            # print(f"  P50:  {latencies.median():.6f} ms")
            # print(f"  P95:  {latencies.quantile(0.95):.6f} ms")
            # print(f"  P99:  {latencies.quantile(0.99):.6f} ms")
            # # ===============================
            # # 6. 内存 after
            # # ===============================
            # mem_after = get_current_memory_mb()
            # peak_after = get_peak_memory_mb()
            #
            # print("\n【空间复杂度（进程法）】")
            # print(
            #     f"  current: {mem_before:.1f} MB → {mem_after:.1f} MB "
            #     f"(+{mem_after - mem_before:.1f} MB)"
            # )
            # print(
            #     f"  peak:    {peak_before:.1f} MB → {peak_after:.1f} MB "
            #     f"(+{peak_after - peak_before:.1f} MB)"
            # )
            #
            # print("\n【GPU 显存】")
            # print(
            #     f"  peak allocated: "
            #     f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB"
            # )
            # print(
            #     f"  peak reserved:  "
            #     f"{torch.cuda.max_memory_reserved() / 1024 / 1024:.1f} MB"
            # )


            return np.mean(net_env.F1), np.mean(net_env.transfer_lag), np.mean(net_env.Reward), np.mean(net_env.bw_use)


def convert(a, b, p, q):
    for i in range(q):
        start_idx = i * p
        end_idx = start_idx + p
        b[:, i] = a[start_idx:end_idx]

if __name__ == '__main__':
    output_dir = f'Results'
    os.makedirs(output_dir, exist_ok=True)
    dataset_map = [4, 7, 10]
    merged_data_map = {}
    seq_start = 0
    val_all_chunks = []
    for h5_path in h5_files:
        dataset_name = os.path.basename(h5_path).replace('2_train_DAO.h5', '')
        try:
            val_chunks = get_seq_chunks_list_by_h5(h5_path)
            if dataset_name == 'DETRAC':
                max_seq = 4
            elif dataset_name == 'DSEC':
                max_seq = 3
            elif dataset_name == 'LMOT':
                max_seq = 3
            else:
                max_seq = 15
            val_all_chunks.extend(val_chunks)

            tmp_df = load_h5_file(h5_path, seq_min=0, seq_max=max_seq - 1)
            data_map = get_chunk_data_map(tmp_df,
                                          seq_min=0,
                                          seq_max=max_seq - 1,
                                          seq_start=seq_start)
            merged_data_map.update(data_map)
            print(f"{dataset_name}: seq范围 [{seq_start}, {seq_start + max_seq - 1}]，共 {len(data_map)} 条。")
            seq_start += max_seq
        except Exception as e:
            print(f"读取 {dataset_name} 时出错: {e}")

    # 大列表，用于汇总每个数据集的时间列表
    all_datasets_times = []
    # 逐个读取文件
    for file in encoding_files:
        # 读取 CSV（假设只有一列是时间）
        df = pd.read_csv(file)
        # 获取时间列（若有多列，可根据列名调整）
        time_values = df.iloc[:, 2].tolist()  # 取第一列作为时间数据
        # 检查长度是否为100
        if len(time_values) != 33:
            print(f"⚠️ 警告：文件 {file} 中的时间数为 {len(time_values)}，不是132！")
        # 添加到大列表中
        all_datasets_times.append(time_values)

    trace_dir = '/home/ubuntu/lyra/CASVA/test_trace/4G/'
    traces = os.listdir(trace_dir)
    for i, trace in enumerate(traces):
        if i not in [4]:
            continue
        SEQ_TOTAL = 60  # 86400 25
        current_video_id = 0
        avg_F1 = 0.0
        avg_Lag = 0.0
        avg_Reward = 0.0
        avg_BW_use = 0.0
        # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
        cooked_bw, cooked_name = load_one_trace(trace_dir, i)
        start = 2
        net_env = DAO_env.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map, seq_chunks=0, start=start, seq_id=0)
        # 认为是有10个视频序列，每个视频序列有180个chunk，每个chunk为2s
        chunk_idx = 0
        for seq_id in range(SEQ_TOTAL):
            if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
                # SEQ_START += dataset_pass_num[current_video_id]
                chunk_idx = 0
                current_video_id += 1
            if current_video_id == 0 or current_video_id == 3:
                net_env.FRAME = FRAMES[0]
            else:
                net_env.FRAME = FRAMES[1]
            video_encoding_time = all_datasets_times[current_video_id]
            net_env.SEQ_CHUNKS = val_all_chunks[seq_id]
            net_env.seq_id = seq_id
            net_env.video_chunk_counter = 0
            net_env.F1 = []
            net_env.transfer_lag = []
            net_env.Reward = []
            net_env.bw_use = []
            F1, lag, reward, bit = test(net_env, video_encoding_time,
                                   images[current_video_id], chunk_idx,
                                   images_xx[current_video_id], RES[current_video_id], current_video_id, seq_id)
            with open(f'{output_dir}/32-48/32-48_{i}.txt', 'a') as f:
                f.write(f'{current_video_id}    {seq_id}    {F1}    {lag}   {reward}    {bit}\n')
                avg_F1 += F1
                avg_Lag += lag
                avg_Reward += reward
                avg_BW_use += bit
            chunk_idx += val_all_chunks[seq_id]
        # with open(f'{output_dir}/log_test.txt', 'a') as f:
        #     f.write(f'avg   {avg_F1 / SEQ_TOTAL}   {avg_Lag / SEQ_TOTAL}   {avg_Reward / SEQ_TOTAL}   {avg_BW_use / SEQ_TOTAL}\n')