import os
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from replay_memory import ReplayMemory
from utils import load_one_trace, get_seq_chunks_list_by_h5, load_trace, load_h5_file, get_chunk_data_map, get_sorted_config_list, load_deg_h5_as_chunk_map
import env
import pandas as pd
import logging
from network import Actor
from Expert import ExpertDP
from test import valid

SKIP = [1.0, 0.75, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]

RANDOM_SEED = 28
LEARNING_RATE = 1e-3
S_INFO = 4
D_INFO = 128
S_LEN = 8
A_DIM = 125

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

SUMMARY_DIR = 'Results/'
LOG_FILE = 'Results/log'

ALL_BW, ALL_NAME = load_trace('/home/dell/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
h5_files = ['/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DETRAC_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DSEC_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/LMOT_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/D²-City_desc.h5',]
encoding_files = ['/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_DETRAC.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_DSEC.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_LMOT.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_D²-City.csv']
deg_files = ['/home/dell/lyra/Can/deg_feats/DETRAC_layer1.h5',
            '/home/dell/lyra/Can/deg_feats/DSEC_layer1.h5',
            '/home/dell/lyra/Can/deg_feats/LMOT_layer1.h5',
            '/home/dell/lyra/Can/deg_feats/D²-City_layer1.h5',]
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
OUTPUT_DIR = "Results/IL"
NAME = "IL"

def train_IL():
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    logging.basicConfig(filename=LOG_FILE + f'/log_central_{NAME}',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + f'/log_test_{NAME}', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        model_IL = Actor().type(dtype)
        model_IL.eval()
        optimizer_IL = optim.Adam(model_IL.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        epoch = 0
        batch_size = 32
        criterion = nn.CrossEntropyLoss()

        SEQ_TOTAL = 0
        dataset_map = [10, 15, 21]
        merged_data_map_train = {}
        merged_data_map_val = {}
        merged_deg_map_train = {}
        merged_deg_map_val = {}
        seq_train_start = 0
        seq_val_start = 0
        val_all_chunks = []  # 全部验证集的块数列表
        train_all_chunks = []  # 全部测试集的块数列表
        train_config_list = []
        val_config_list = []
        for idx, h5_path in enumerate(h5_files):
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

                SEQ_TOTAL += train_num
                start_idx = max_seq - sum_seq
                val_chunks = seq_chunks_list[start_idx + train_num:max_seq]
                train_chunks = seq_chunks_list[start_idx:start_idx + train_num]
                val_all_chunks.extend(val_chunks)
                train_all_chunks.extend(train_chunks)

                all_config_list = get_sorted_config_list(h5_path, seq_min=start_idx, seq_max=max_seq - 1)
                train_config_list.extend(all_config_list[:train_num])
                val_config_list.extend(all_config_list[train_num:])

                # 调用函数并传入起始编号
                tmp_df = load_h5_file(h5_path, seq_min=start_idx, seq_max=max_seq - 1)

                data_map = get_chunk_data_map(tmp_df,
                                              seq_min=start_idx,
                                              seq_max=start_idx + train_num - 1,
                                              seq_start=seq_train_start)
                merged_data_map_train.update(data_map)
                print(f"train_{dataset_name}: chunks num {train_all_chunks}")
                print(f"train_{dataset_name}: seq范围 [{seq_train_start}, {seq_train_start + train_num - 1}]，共 {len(data_map)} 条。")
                data_map = get_chunk_data_map(tmp_df,
                                              seq_min=start_idx + train_num,
                                              seq_max=max_seq - 1,
                                              seq_start=seq_val_start)
                merged_data_map_val.update(data_map)
                print(f"val_{dataset_name}: chunks num {val_all_chunks}")
                print(f"val_{dataset_name}: seq范围 [{seq_val_start}, {seq_val_start + (max_seq - train_num - 1)}]，共 {len(data_map)} 条。")
                # ===== 新增：接 deg h5 =====
                deg_h5_path = deg_files[idx]
                deg_train, deg_val, _, _ = load_deg_h5_as_chunk_map(
                    deg_h5_path,
                    seq_train_start=seq_train_start,
                    seq_val_start=seq_val_start,
                    train_num=train_num
                )
                merged_deg_map_train.update(deg_train)
                merged_deg_map_val.update(deg_val)

                seq_train_start += train_num
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

        memory = ReplayMemory(70)
        expert = ExpertDP()

        # seq_chunk_acc, seq_chunk_ul = [], []
        # for seq in range(SEQ_TOTAL):
        #     chunk_nums = seq_chunks_list[seq]
        #     acc, ul = get_acc_ul_tb(seq, chunk_nums, ALL_BW[0])
        #     seq_chunk_acc.append(acc)
        #     seq_chunk_ul.append(ul)
        # rule_agent = lagent.LyapunovAgent()

        while True:
            net_env = env.Environment(ALL_BW[0], merged_data_map_train)
            state = np.zeros((S_INFO, S_LEN))
            state = torch.from_numpy(state)
            current_video_id = 0
            for seq_id in range(SEQ_TOTAL):
                if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
                    current_video_id += 1
                if current_video_id == 0 or current_video_id == 3:
                    net_env.FRAME = FRAMES[0]
                else:
                    net_env.FRAME = FRAMES[1]
                video_encoding_time = all_datasets_times[current_video_id]
                net_env.SEQ_CHUNKS = train_all_chunks[seq_id]
                net_env.SEQ_ID = seq_id
                net_env.video_chunk_counter = 0
                seq_config = train_config_list[seq_id]

                states = []
                deg_states = []
                # actions = []
                exp_actions = []
                seq_ids = []
                chunk_ids = []

                deg_state = np.zeros((S_LEN, D_INFO))
                deg_state = torch.from_numpy(deg_state)
                end_of_video = False

                while not end_of_video:
                    states.append(state.unsqueeze(0))
                    deg_feat = merged_deg_map_train[(net_env.SEQ_ID, net_env.video_chunk_counter)]
                    assert deg_feat.shape[0] == D_INFO
                    deg_state = np.roll(deg_state, -1, axis=0)
                    deg_state[-1,:] = deg_feat
                    deg_state = torch.from_numpy(deg_state)
                    deg_states.append(deg_state.unsqueeze(0))
                    # IL action
                    with torch.no_grad():
                        prob, _ = model_IL(state.unsqueeze(0).type(dtype), deg_state.unsqueeze(0).type(dtype))
                        action = prob.multinomial(num_samples=1).detach()
                        knob = int(action.squeeze().cpu().numpy())

                    qp = knob // 25 # 0 to 10, because 80 // 16 = 5
                    remainder = knob % 25
                    skip = remainder // 5  # 0 to 3
                    re = remainder % 5  # 0 to 3

                    # expert action
                    expert_action, _ = expert.solve(net_env, seq_config[net_env.video_chunk_counter], video_encoding_time)
                    # expert_action = rule_agent.genetic_algorithm_optimization(net_env, Q, video_encoding_time)

                    exp_actions.append(torch.tensor([expert_action]))
                    seq_ids.append(torch.tensor([net_env.SEQ_ID]))
                    chunk_ids.append(torch.tensor([net_env.video_chunk_counter]))

                    et = video_encoding_time[knob]
                    # Execute the action and collect data
                    bw, latency, _, size, f1, end_of_video = net_env.get_video_chunk(qp, skip, re, et)

                    # actions.append(torch.tensor([action]))


                    state = np.roll(state, -1, axis=1)
                    state[0, -1] = bw
                    state[3, -1] = qp
                    state[4, -1] = skip
                    state[5, -1] = re
                    state = torch.from_numpy(state)
                # memory.push([states, actions, f1s, seq_ids, chunk_ids])
                video = {
                    'states': torch.cat(states, dim=0),  # [T, S_INFO, S_LEN]
                    'deg_states': torch.cat(deg_states, dim=0), # [T, S_LEN, D_INFO]
                    # 'actions': torch.cat(actions, dim=0),  # [T]
                    'exp_actions': torch.cat(exp_actions, dim=0),  # [T]
                    'seq_ids': torch.cat(seq_ids, dim=0),  # [T]
                    'chunk_ids': torch.cat(chunk_ids, dim=0)  # [T]
                }
                memory.push_video(video)
            # train
            model_IL.train()
            optimizer_IL.zero_grad()
            videos = memory.sample_videos_by_chunk_budget(batch_size)
            video_losses = []
            for video in videos:
                states = video['states'].type(dtype)
                deg_states = video['deg_states'].type(dtype)
                # actions = video['actions'].long().to(states.device)
                exp_actions = video['exp_actions'].long().to(states.device)
                outputs, logits = model_IL(states, deg_states)
                loss = criterion(logits, exp_actions)
                video_losses.append(loss)
            loss = torch.stack(video_losses).mean()

            # batch_states, batch_actions, batch_f1, batch_seq_id, batch_chunk_id = memory.sample(batch_size)
            # outputs, predictions = model_IL(batch_states.type(dtype))
            # actions = outputs.multinomial(num_samples=1).detach().squeeze()
            # loss = criterion(predictions, batch_actions.type(torch.long).to(predictions.device))
            # actions = actions.cpu().numpy()
            loss.backward()
            optimizer_IL.step()
            epoch += 1
            memory.clear()
            if epoch % 10 == 0:
                valid(model_IL, epoch, test_log_file, val_all_chunks, merged_data_map_val, merged_deg_map_val, all_datasets_times)
                # print(f'Epoch: {epoch}, Loss: {loss.item()}, F1: {mean_f1}, PRED_F1: {pre_f1}')
                # if not os.path.exists(OUTPUT_DIR):
                #     os.makedirs(OUTPUT_DIR)
                # model_path = f'{OUTPUT_DIR}/IL_{epoch}.model'
                # torch.save(model_IL.state_dict(), model_path)
            if epoch > 200:
                break


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_IL()