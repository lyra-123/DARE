import os
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from replay_memory import ReplayMemory
from utils import load_trace, load_one_trace, get_seq_chunks_list_by_h5, get_chunk_data_map
from IL import ILAgent, A_DIM
import lagent
import env
import pandas as pd
import math
import cProfile, pstats, io

#LMOT、DSEC
# QP = [1, 1.2778, 1.5556, 1.8333, 2.1111]
# D²-City_1080p、DETRAC、D²-City_720p
# QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]

# D²-City_1080p、DETRAC、D²-City_720p
# SKIP = [1.0, 0.5, 0.3333, 0.1667]
# LMOT、DSEC
# FPS = [1.0, 0.5, 0.3333, 0.2]

# DETRAC
# RE = [1.0, 0.7907, 0.4444, 0.1972] # 1280*720/1920*1080
# LMOT
# RE = [1.0, 0.5184, 0.36, 0.2304]
# DSEC
# RE = [1, 0.5625, 0.4444, 0.25, 0.1111]
# D²-City_1080p
# RE = [1, 0.4444, 0.25, 0.1667, 0.037]
# D²-City_720p
# RE = [1, 0.5625, 0.4448, 0.1109]

SKIP = [1.0, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]

RANDOM_SEED = 28
LEARNING_RATE = 1e-3
# 专家学习结果与IL自学习结果差异的阈值
THRESHOLD = 0.3
S_INFO = 6
S_LEN = 8

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/dell/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
h5_files = ['/home/dell/lyra/CASVA/h5_file/DETRAC_train.h5',
            '/home/dell/lyra/CASVA/h5_file/DSEC_train.h5',
            '/home/dell/lyra/CASVA/h5_file/LMOT_train.h5',
            '/home/dell/lyra/CASVA/h5_file/D²-City_train.h5',
            ]
encoding_files = ['/home/dell/lyra/CASVA/coding_time/coding_time_DETRAC.csv',
            '/home/dell/lyra/CASVA/coding_time/coding_time_DSEC.csv',
            '/home/dell/lyra/CASVA/coding_time/coding_time_LMOT.csv',
            '/home/dell/lyra/CASVA/coding_time/coding_time_D²-City.csv',
            ]
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
OUTPUT_DIR = "Results/IL"
NAME = "IL"

def cal_upload_lag(cooked_bw,video_chunk_size):
    end = 2
    RTT = 0.08
    # 模拟传输，cooked_bw是以秒为单位记录的
    while True:
        if math.ceil(end) == end:
            if end + 1 >= len(cooked_bw):
                real_bw = cooked_bw[int(end + 1) % len(cooked_bw)]
            else:
                real_bw = cooked_bw[int(end + 1)]
            duration = 1

        else:
            if math.ceil(end) >= len(cooked_bw):
                real_bw = cooked_bw[math.ceil(end) % len(cooked_bw)]
            else:
                real_bw = cooked_bw[math.ceil(end)]
            duration = math.ceil(end) - end

        if video_chunk_size - real_bw * 1000 * duration >= 0:
            video_chunk_size = video_chunk_size - real_bw * 1000 * duration
            end += duration
        else:
            end += video_chunk_size / (real_bw * 1000)
            video_chunk_size = 0

        if video_chunk_size == 0:
            upload_lag = end - 2 + RTT
            break
    return upload_lag

def get_acc_ul_tb(seq_id,chunk_nums,cooked_trace):
    acc = np.zeros((chunk_nums, A_DIM))
    ul = np.zeros((chunk_nums, A_DIM))
    for i in range(chunk_nums):
        for j in range(A_DIM):
            qp = j // 20  # 0 to 10, because 80 // 16 = 5
            remainder = j % 20
            s = remainder // 5  # 0 to 3
            r = remainder % 5  # 0 to 3
            acc[i, j] = df.loc[(seq_id, i, qp, s, r), 'Accuracy']
            chunk_size = df.loc[(seq_id, i, qp, s, r), 'Size']
            ul[i, j] = cal_upload_lag(cooked_trace, chunk_size)
    return acc,ul

def train_IL():
    torch.manual_seed(RANDOM_SEED)

    model_IL = ILAgent().type(dtype)
    model_IL.eval()
    optimizer_IL = optim.Adam(model_IL.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    epoch = 0
    batch_size = 32
    criterion = nn.CrossEntropyLoss()

    SEQ_TOTAL = 0
    # seq_chunks_list = get_seq_chunks_list(237)
    # seq_chunks_list = get_seq_chunks_list_by_h5(h5_file)
    # seq_chunk_data = get_chunk_data_map(h5_file)
    dataset_map = [20, 31, 39]
    dataset_pass_num = [7, 9, 3]
    merged_data_map = {}
    seq_start = 0
    test_all_chunks = []  # 全部测试集的块数列表
    for h5_path in h5_files:
        dataset_name = os.path.basename(h5_path).replace('_train.h5', '')
        try:
            seq_chunks_list = get_seq_chunks_list_by_h5(h5_path)
            if dataset_name == 'DETRAC':
                test_num = 20
                max_seq = 26
            elif dataset_name == 'DSEC':
                test_num = 11
                max_seq = 19
            elif dataset_name == 'LMOT':
                test_num = 8
                max_seq = 10
            else:
                test_num = 73
                max_seq = 99
            SEQ_TOTAL += test_num

            test_chunks = seq_chunks_list[:test_num]
            test_all_chunks.extend(test_chunks)

            # 调用函数并传入起始编号
            data_map = get_chunk_data_map(h5_path, seq_min=0, seq_max=max_seq, seq_start=seq_start)
            merged_data_map.update(data_map)
            print(f"{dataset_name}: seq范围 [{seq_start}, {seq_start + max_seq}]，共 {len(data_map)} 条。")
            seq_start += max_seq + 1
        except Exception as e:
            print(f"读取 {dataset_name} 时出错: {e}")

    # 大列表，用于汇总每个数据集的时间列表
    all_datasets_times = []
    # 逐个读取文件
    for file in encoding_files:
        # 读取 CSV（假设只有一列是时间）
        df = pd.read_csv(file)
        # 获取时间列（若有多列，可根据列名调整）
        time_values = df.iloc[:, 3].tolist()  # 取第一列作为时间数据
        # 检查长度是否为100
        if len(time_values) != 100:
            print(f"⚠️ 警告：文件 {file} 中的时间数为 {len(time_values)}，不是100！")
        # 添加到大列表中
        all_datasets_times.append(time_values)

    memory = ReplayMemory(SEQ_TOTAL * 65)

    # seq_chunk_acc, seq_chunk_ul = [], []
    # for seq in range(SEQ_TOTAL):
    #     chunk_nums = seq_chunks_list[seq]
    #     acc, ul = get_acc_ul_tb(seq, chunk_nums, ALL_BW[0])
    #     seq_chunk_acc.append(acc)
    #     seq_chunk_ul.append(ul)
    rule_agent = lagent.LyapunovAgent()

    while True:
        net_env = env.Environment(ALL_BW[0], merged_data_map)
        current_video_id = 0
        pass_num = 0
        for seq_id in range(SEQ_TOTAL):
            if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
                pass_num += dataset_pass_num[current_video_id]
                current_video_id += 1
            if current_video_id == 0 or current_video_id == 3:
                net_env.FRAME = FRAMES[0]
            else:
                net_env.FRAME = FRAMES[1]
            video_encoding_time = all_datasets_times[current_video_id]
            net_env.SEQ_CHUNKS = test_all_chunks[seq_id]
            net_env.SEQ_ID = seq_id + pass_num
            net_env.video_chunk_counter = 0

            state = np.zeros((S_INFO, S_LEN))
            state = torch.from_numpy(state)

            states = []
            actions = []
            f1s = []
            seq_ids = []
            chunk_ids = []

            Q = 0
            end_of_video = False

            while not end_of_video:
                states.append(state.unsqueeze(0))
                # IL action
                with torch.no_grad():
                    prob, _ = model_IL(state.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    knob = int(action.squeeze().cpu().numpy())

                qp = knob // 20  # 0 to 10, because 80 // 16 = 5
                remainder = knob % 20
                skip = remainder // 5  # 0 to 3
                re = remainder % 5  # 0 to 3
                IL_action = (re, skip, qp)

                # expert action
                # expert_action = rule_agent.genetic_algorithm_optimization(net_env.video_chunk_counter, chunk_acc, chunk_ul, Q)
                expert_action = rule_agent.genetic_algorithm_optimization(net_env, Q, video_encoding_time)

                # Compare actions
                action = compare(IL_action, expert_action)
                # actions.append(torch.tensor([action[2] * 16 + action[1] * 4 + action[0]]))
                seq_ids.append(torch.tensor([net_env.SEQ_ID]))
                chunk_ids.append(torch.tensor([net_env.video_chunk_counter]))
                knob = action[2] * 20 + action[1] * 5 + action[0]
                et = video_encoding_time[knob]

                # Execute the action and collect data
                bw, latency, _, f1, Q, _, end_of_video = net_env.get_video_chunk(action[2], action[1], action[0], et)

                actions.append(torch.tensor([knob]))
                f1s.append(torch.tensor([f1]))

                state = np.roll(state, -1, axis=1)
                state[0, -1] = bw
                state[1, -1] = latency
                state[2, -1] = Q
                state[3, -1] = qp
                state[4, -1] = skip
                state[5, -1] = re
                state = torch.from_numpy(state)
            memory.push([states, actions, f1s, seq_ids, chunk_ids])
        # train
        model_IL.train()
        optimizer_IL.zero_grad()
        batch_states, batch_actions, batch_f1, batch_seq_id, batch_chunk_id = memory.sample(batch_size)

        outputs, predictions = model_IL(batch_states.type(dtype))
        actions = outputs.multinomial(num_samples=1).detach().squeeze()
        loss = criterion(predictions, batch_actions.type(torch.long).to(predictions.device))
        actions = actions.cpu().numpy()
        loss.backward()
        optimizer_IL.step()
        epoch += 1
        memory.clear()
        if epoch % 100 == 0:
            pred_f1 = []
            batch_seq_id = batch_seq_id.type(torch.int)
            batch_chunk_id = batch_chunk_id.type(torch.int)
            batch_f1 = batch_f1.type(torch.float)
            # 计算均值
            mean_f1 = batch_f1.mean()  # 计算整个 batch_actions 的均值
            # print("check predictions ",predictions)
            for idx, pred_action in enumerate(actions):
                # qp = pred_action // 20  # 0 to 10, because 80 // 16 = 5
                # remainder = pred_action % 20
                # skip = remainder // 5  # 0 to 3
                # re = remainder % 5  # 0 to 3
                # pred_f = df.loc[(batch_seq_id[idx].item(), batch_chunk_id[idx].item(), qp, skip, re), 'Accuracy']
                _, pred_f, _ = merged_data_map[(batch_seq_id[idx].item(), batch_chunk_id[idx].item(), pred_action)]
                pred_f1.append(pred_f)
            pre_f1 = np.mean(pred_f1)

            print(f'Epoch: {epoch}, Loss: {loss.item()}, F1: {mean_f1}, PRED_F1: {pre_f1}' )
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            model_path = f'{OUTPUT_DIR}/LCA/IL_{epoch}.model'
            torch.save(model_IL.state_dict(), model_path)
        if epoch > 20000:
            break

def compare(IL_action, expert_action, threshold=THRESHOLD):
    a = (RE[IL_action[0]], SKIP[IL_action[1]], QP[IL_action[2]])
    b = (RE[expert_action[0]], SKIP[expert_action[1]], QP[expert_action[2]])

    IL_vector = torch.tensor(a, dtype=torch.float32)
    expert_vector = torch.tensor(b, dtype=torch.float32)

    if torch.norm(IL_vector - expert_vector) <= threshold:
        return IL_action
    else:
        return expert_action


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_IL()