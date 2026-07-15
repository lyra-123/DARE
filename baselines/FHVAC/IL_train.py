import os
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from replay_memory import ReplayMemory
from utils import load_trace, get_seq_chunks_list_by_h5, load_one_trace, get_chunk_data_map, load_h5_file
from IL import ILAgent
from Rule_Based import rule_based
import env
import pandas as pd
from test import valid_IL

FPS = [1.0, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]

RANDOM_SEED = 28
LEARNING_RATE = 1e-3
# 专家学习结果与IL自学习结果差异的阈值
THRESHOLD = 0.3
S_INFO = 7
# 考虑pt
# S_INFO = 8
S_LEN = 8

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/ubuntu/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
# h5_file = '/home/dell/lyra/CASVA/h5_file/FHVAC-CASVA/D²-City_720p_train.h5'
h5_files = ['/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/DETRAC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/DSEC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/LMOT_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/D²-City_desc.h5',]
encoding_files = ['/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DETRAC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DSEC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_LMOT.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_D²-City.csv']
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
OUTPUT_DIR = "Results/IL"
NAME = "IL"

def train_IL():
    torch.manual_seed(RANDOM_SEED)

    model_IL = ILAgent().type(dtype)
    model_IL.train()
    model_IL.load_state_dict(torch.load('/home/ubuntu/lyra/FHVAC/Results/IL/IL_20000.model', weights_only=True))
    optimizer_IL = optim.Adam(model_IL.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    epoch = 0
    batch_size = 32
    criterion = nn.CrossEntropyLoss()

    SEQ_TOTAL = 0
    # seq_chunks_list = get_seq_chunks_list(11)
    # seq_chunks_list = get_seq_chunks_list_by_h5(h5_file)

    dataset_map = [10, 15, 21]
    dataset_pass_num = [7, 9, 3]
    merged_data_map_train = {}
    merged_data_map_val = {}
    seq_train_start = 0
    seq_val_start = 0
    val_all_chunks = []  # 全部验证集的块数列表
    train_all_chunks = []  # 全部测试集的块数列表
    for h5_path in h5_files:
        dataset_name = os.path.basename(h5_path).replace('_desc.h5', '')
        try:
            seq_chunks_list = get_seq_chunks_list_by_h5(h5_path)
            if dataset_name == 'DETRAC':
                sum_seq = 14
                train_num = 10
                max_seq = 14
            elif dataset_name == 'DSEC':
                sum_seq = 8
                train_num = 5
                max_seq = 8
            elif dataset_name == 'LMOT':
                sum_seq = 9
                train_num = 6
                max_seq = 9
            else:
                sum_seq = 55
                train_num = 40
                max_seq = 106

            start_idx = max_seq - sum_seq
            # val_chunks = seq_chunks_list[train_num:max_seq]
            # train_chunks = seq_chunks_list[:train_num]
            val_chunks = seq_chunks_list[start_idx + train_num:max_seq]
            train_chunks = seq_chunks_list[start_idx:start_idx + train_num]

            SEQ_TOTAL += train_num
            val_all_chunks.extend(val_chunks)
            train_all_chunks.extend(train_chunks)

            # 调用函数并传入起始编号
            tmp_df = load_h5_file(h5_path, seq_min=start_idx, seq_max=max_seq - 1)

            data_map = get_chunk_data_map(tmp_df,
                                          seq_min=start_idx,
                                          seq_max=start_idx + train_num - 1,
                                          seq_start=seq_train_start)
            merged_data_map_train.update(data_map)
            print(f"train_{dataset_name}: chunks num {train_all_chunks}")
            print(
                f"train_{dataset_name}: seq范围 [{seq_train_start}, {seq_train_start + train_num - 1}]，共 {len(data_map)} 条。")
            seq_train_start += train_num

            data_map = get_chunk_data_map(tmp_df,
                                          seq_min=start_idx + train_num,
                                          seq_max=max_seq - 1,
                                          seq_start=seq_val_start)
            merged_data_map_val.update(data_map)
            print(f"val_{dataset_name}: chunks num {val_all_chunks}")
            print(
                f"val_{dataset_name}: seq范围 [{seq_val_start}, {seq_val_start + (max_seq - train_num - 1)}]，共 {len(data_map)} 条。")
            seq_val_start += (max_seq - train_num)
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

    memory = ReplayMemory(1300)
    # df = pd.read_hdf(h5_file, 'encoding_data')
    valid_IL(model_IL, val_all_chunks, merged_data_map_val, all_datasets_times)

    # while True:
    #     net_env = env.Environment(ALL_BW[0], merged_data_map_train)
    #     state = np.zeros((S_INFO, S_LEN))
    #     state = torch.from_numpy(state)
    #     Q = 0
    #     bw_est = 0
    #     current_video_id = 0
    #     pass_num = 0
    #     for seq_id in range(SEQ_TOTAL):
    #         if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
    #             # pass_num += dataset_pass_num[current_video_id]
    #             current_video_id += 1
    #         if current_video_id == 0 or current_video_id == 3:
    #             net_env.FRAME = FRAMES[0]
    #         else:
    #             net_env.FRAME = FRAMES[1]
    #         video_encoding_time = all_datasets_times[current_video_id]
    #         net_env.SEQ_CHUNKS = train_all_chunks[seq_id]
    #         net_env.SEQ_ID = seq_id
    #         net_env.video_chunk_counter = 0
    #
    #         states = []
    #         actions = []
    #         f1s = []
    #         seq_ids = []
    #         chunk_ids = []
    #
    #         end_of_video = False
    #
    #         while not end_of_video:
    #             states.append(state.unsqueeze(0))
    #             # IL action
    #             prob, _ = model_IL(state.unsqueeze(0).type(dtype))
    #             action = prob.multinomial(num_samples=1).detach()
    #             knob = int(action.squeeze().cpu().numpy())
    #
    #             qp = knob // 20  # 0 to 10, because 80 // 16 = 5
    #             remainder = knob % 20
    #             skip = remainder // 5  # 0 to 3
    #             re = remainder % 5  # 0 to 3
    #             IL_action = (re, skip, qp)
    #
    #             # expert action
    #             expert_action = rule_based(Q, bw_est, current_video_id)
    #
    #             # Compare actions
    #             action = compare(IL_action, expert_action)
    #             # actions.append(torch.tensor([action[2] * 16 + action[1] * 4 + action[0]]))
    #             seq_ids.append(torch.tensor([net_env.SEQ_ID]))
    #             chunk_ids.append(torch.tensor([net_env.video_chunk_counter]))
    #             knob = action[2] * 20 + action[1] * 5 + action[0]
    #             et = video_encoding_time[knob]
    #
    #             # Execute the action and collect data
    #             bw_est, bw, latency, buffer_size, size, dynamics, f1, Q, end_of_video = net_env.get_video_chunk(action[2], action[1], action[0], et)
    #             # 考虑 pt
    #             # bw_est, bw, latency, buffer_size, size, dynamics, f, Q, P, end_of_video = net_env.get_video_chunk(
    #             #     action[2], action[1], action[0])
    #
    #             actions.append(torch.tensor([knob]))
    #             f1s.append(torch.tensor([f1]))
    #
    #             state = np.roll(state, -1, axis=1)
    #             state[0, -1] = bw
    #             state[1, -1] = latency
    #             state[2, -1] = buffer_size
    #             state[3, -1] = qp
    #             state[4, -1] = skip
    #             state[5, -1] = re
    #             state[6, -1] = Q
    #             # 考虑 pt
    #             # state[7, -1] = P
    #             state = torch.from_numpy(state)
    #         memory.push([states, actions, f1s, seq_ids, chunk_ids])
    #     # train
    #     optimizer_IL.zero_grad()
    #     batch_states, batch_actions, batch_f1, batch_seq_id, batch_chunk_id = memory.sample(batch_size)
    #
    #     outputs, predictions = model_IL(batch_states.type(dtype))
    #     actions = outputs.multinomial(num_samples=1).detach().squeeze()
    #     loss = criterion(predictions, batch_actions.type(torch.long).to(predictions.device))
    #     actions = actions.cpu().numpy()
    #     loss.backward()
    #     optimizer_IL.step()
    #     epoch += 1
    #     memory.clear()
    #     if epoch % 100 == 0:
    #         pred_f1 = []
    #         batch_seq_id = batch_seq_id.type(torch.int)
    #         batch_chunk_id = batch_chunk_id.type(torch.int)
    #         batch_f1 = batch_f1.type(torch.float)
    #         # 计算均值
    #         mean_f1 = batch_f1.mean()  # 计算整个 batch_actions 的均值
    #         # print("check predictions ",predictions)
    #         for idx, pred_action in enumerate(actions):
    #             _, pred_f, _ = merged_data_map_train[(batch_seq_id[idx].item(), batch_chunk_id[idx].item(), pred_action)]
    #             pred_f1.append(pred_f)
    #         pre_f1 = np.mean(pred_f1)
    #
    #         print(f'Epoch: {epoch}, Loss: {loss.item()}, F1: {mean_f1}, PRED_F1: {pre_f1}' )
    #         if not os.path.exists(OUTPUT_DIR):
    #             os.makedirs(OUTPUT_DIR)
    #         model_path = f'{OUTPUT_DIR}/IL_{epoch}.model'
    #         torch.save(model_IL.state_dict(), model_path)
    #     if epoch > 20000:
    #         break

def compare(IL_action, expert_action, threshold=THRESHOLD):
    a = (RE[IL_action[0]], FPS[IL_action[1]], QP[IL_action[2]])
    b = (RE[expert_action[0]], FPS[expert_action[1]], QP[expert_action[2]])

    IL_vector = torch.tensor(a, dtype=torch.float32)
    expert_vector = torch.tensor(b, dtype=torch.float32)

    if torch.norm(IL_vector - expert_vector) <= threshold:
        return IL_action
    else:
        return expert_action


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_IL()