from utils import load_one_trace
import raw_env
import pandas as pd
from collections import defaultdict
import numpy as np

NAME = 'd2'
csv_file = "/home/ubuntu/lyra/MPC/h5_file/raw_acc_size_bitrate_d2.csv"   # 改成你的 CSV 路径
# 读取 CSV
df = pd.read_csv(csv_file)
# ============================
# 1. 有多少个 Seq
# ============================
seq_values  = df["Seq"].unique()
SEQ_TOTAL = len(seq_values)
# ============================
# 2. 每个 Seq 有多少个 Chunk
# ============================
chunks_per_seq = df.groupby("Seq")["Chunk"].nunique().to_dict()
# 转成列表（按 Seq 排序）
val_all_chunks = [chunks_per_seq[s] for s in sorted(chunks_per_seq)]

# 找最大 seq 和最大 chunk，用来定数组大小
max_seq = df["Seq"].max()
max_chunk = df["Chunk"].max()
# 初始化二维数组
table = [[None for _ in range(max_chunk + 1)] for _ in range(max_seq + 1)]
# 填充
for _, row in df.iterrows():
    seq = int(row["Seq"])
    chunk = int(row["Chunk"])
    acc = float(row["Accuracy"])
    size = int(row["Size"])
    bit = int(row["Bitrate"])
    table[seq][chunk] = (acc, size, bit)

cooked_bw, cooked_name = load_one_trace('/home/ubuntu/lyra/CASVA/test_trace/4G/', 4)
env = raw_env.Environment(cooked_bw=cooked_bw, seq_chunk_data=table, seq_chunks=0, start=2)
f1_sum, lag_sum, reward_sum, bit_sum = 0., 0., 0., 0.
for seq_id in range(SEQ_TOTAL):
    env.F1 = []
    env.lag = []
    env.Reward = []
    env.bw_use = []
    env.SEQ_CHUNKS = val_all_chunks[seq_id]
    env.seq_id = seq_id
    env.video_chunk_counter = 0
    end_of_video = False
    while not end_of_video:
        latency, f1, end_of_video = env.get_video_chunk()
    f1 = np.mean(env.F1)
    lag = np.mean(env.lag)
    reward = np.mean(env.Reward)
    bit = np.mean(env.bw_use)
    f1_sum += f1
    lag_sum += lag
    reward_sum += reward
    bit_sum += bit
    with open(f'Results/test_results/raw_{NAME}_log_test.txt', 'a') as f:
        f.write(f'{seq_id}    {f1}    {lag}   {reward}    {bit}\n')

f1_avg = f1_sum / SEQ_TOTAL
lag_avg = lag_sum / SEQ_TOTAL
reward_avg = reward_sum / SEQ_TOTAL
bit_avg = bit_sum / SEQ_TOTAL
with open(f'Results/test_results/raw_{NAME}_log_test.txt', 'a') as f:
    f.write(f'avg   {f1_avg}   {lag_avg}   {reward_avg}    {bit_avg}\n')