import os
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Process, Queue, set_start_method
from torch.autograd import Variable
import logging
from utils import load_trace, C_R, get_seq_chunks_list_by_h5, load_one_trace, get_chunk_data_map, load_h5_file
from PPO import Actor, Critic
from replay_memory import ReplayMemory
from test import valid
import env
import pandas as pd


# --------------------------------
A_DIM = 100  # 动作维度，qp可选数量 * re可选数量 * fps可选数量 = 5 * 4 * 4 = 80
# --------------------------------
RANDOM_SEED = 28
S_INFO = 8  # bw, delay, buffer, qp, skip, r, segment_size, dynamics
S_LEN = 8  # past 8
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
UPDATE_INTERVAL = 100
L = 2
# CHUNK_NUM = len(os.listdir("/home/dell/lyra/CASVA/dataset/video_DETRAC_train"))

SUMMARY_DIR = 'Results/'
LOG_FILE = 'Results/log'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/ubuntu/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
video_file = '/home/dell/lyra/CASVA/dataset/video_LMOT_sort_CASVA_train'
h5_files = ['/home/ubuntu/lyra/CASVA/h5_file/desc/DETRAC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/desc/DSEC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/desc/LMOT_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/desc/D²-City_desc.h5',]
encoding_files = ['/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DETRAC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DSEC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_LMOT.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_D²-City.csv']
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
# h5_file = '/home/dell/lyra/CASVA/h5_file/FHVAC-CASVA/D²-City_720p_train.h5'
NAME = 'RL'


def get_seq_chunks_list(SEQ):
    chunk_list = sorted(os.listdir(video_file))
    seq_chunks = {seq_id: 0 for seq_id in range(SEQ)}
    for chunk in chunk_list:
        seq_id = int(chunk.split('_')[0])
        seq_chunks[seq_id] += 1
    return seq_chunks

def train_ppo():
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    logging.basicConfig(filename=LOG_FILE + f'/log_central_{NAME}',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + f'/log_test_{NAME}', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)

        model_actor = Actor().type(dtype)
        model_critic = Critic().type(dtype)

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.Adam(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.Adam(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

        # 总视频序列
        VIDEO_TOTAL = 100
        exploration_size = 0
        # 其实意味着要处理1800个chunk，因为我们这里总共只有480个chunk，所以可以设置为480
        # episode_steps = get_seq_chunks_list(VIDEO_TOTAL) # 1800
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

                exploration_size += train_num
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

        # episode_steps = get_seq_chunks_list_by_h5(h5_file)
        # last_knob = 40
        update_num = 2
        batch_size = 256 # 256
        gamma = 0.99
        gae_param = 0.97
        c = 3
        clip = 0.2
        ent_coeff = 0.9
        memory = ReplayMemory(1300)

        epoch = 0

        # 其实隐含了训练 50000 epoches
        while True:
            net_env = env.Environment(ALL_BW[0], merged_data_map_train)
            state = np.zeros((S_INFO, S_LEN))
            state = torch.from_numpy(state)
            current_video_id = 0
            pass_num = 0
            for explore in range(exploration_size):
                # net_env = env.Environment(ALL_BW[0])
                # net_env.SEQ_CHUNKS = episode_steps[explore]
                # net_env.SEQ_ID = explore
                if current_video_id < 3 and explore == dataset_map[current_video_id]:
                    # pass_num += dataset_pass_num[current_video_id]
                    current_video_id += 1
                if current_video_id == 0 or current_video_id == 3:
                    net_env.FRAME = FRAMES[0]
                else:
                    net_env.FRAME = FRAMES[1]
                video_encoding_time = all_datasets_times[current_video_id]
                net_env.SEQ_CHUNKS = train_all_chunks[explore]
                net_env.SEQ_ID = explore
                net_env.video_chunk_counter = 0
                last_buff = 0

                states = []
                actions = []
                rewards = []
                values = []
                returns = []
                advantages = []
                end_of_video = False

                # 每个epoch会执行CHUNK_NUM次actor决策
                while not end_of_video:
                    prob = model_actor(state.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()

                    values.append(v)
                    knob = int(action.squeeze().cpu().numpy())
                    actions.append(torch.tensor([action]))
                    states.append(state.unsqueeze(0))

                    qp = knob // 20  # 0 to 10, because 80 // 16 = 5
                    remainder = knob % 20
                    skip = remainder // 5  # 0 to 3
                    re = remainder % 5  # 0 to 3
                    et = video_encoding_time[knob]

                    _, bw, latency, buffer_size, size, dynamics, f1, _, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
                    # reward = 2 * f1 - 1.5 * max((latency-2), 0) + 0.5 * max(last_buff - buffer_size, 0)
                    reward = 4 * f1 - 1.5 * latency
                    # reward = 2 * f1 - latency
                    # reward = C_R(f1, latency)
                    rewards.append(reward)

                    last_buff = buffer_size
                    # last_knob = knob

                    state = np.roll(state, -1, axis=1)
                    state[0, -1] = bw
                    state[1, -1] = latency
                    state[2, -1] = buffer_size
                    state[3, -1] = qp
                    state[4, -1] = skip
                    state[5, -1] = re
                    state[6, -1] = size
                    state[7, -1] = dynamics
                    state = torch.from_numpy(state)

                # 如果episode_steps的值小于视频的总块数，那么跳出循环的条件就不是通过end_of_video，此时end_of_video的值就为False
                R = torch.zeros(1, 1)
                if not end_of_video:
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                    R = v.data
                # ================================结束一个ep========================================

                # 这里的v.data是从评论家网络输出的值，v.data是一个没有梯度信息的张量
                # Variable(v.data)会将其转换为一个可以进行反向传播的Variable对象
                values.append(Variable(R))
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    A = float(td) + gamma * gae_param * A
                    advantages.append(A)
                    R = A + values[i]
                    returns.append(R)
                advantages.reverse()
                returns.reverse()
                memory.push([states, actions, returns, advantages])
            model_actor_old = Actor().type(dtype)
            model_actor_old.load_state_dict(model_actor.state_dict())
            model_critic_old = Critic().type(dtype)
            model_critic_old.load_state_dict(model_critic.state_dict())

            for update_step in range(update_num):
                model_actor.zero_grad()
                model_critic.zero_grad()

                batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)

                # --------------------------------------------------------------------------------
                # Calculate policy loss
                probs_old = model_actor_old(batch_states.type(dtype).detach())
                probs_new = model_actor(batch_states.type(dtype))
                ratio = calculate_prob_ratio(probs_new, probs_old, batch_actions)

                advantages = batch_advantages.type(dtype)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
                loss_api = -torch.mean(torch.min(surr1, surr2))

                entropy = calculate_entropy(probs_new)
                loss_ent = -ent_coeff * entropy
                total_loss_api = loss_api + loss_ent
                # -----------------------------------------------------------------------------------

                # Update critic networks
                v_pre = model_critic(batch_states.type(dtype))
                v_pre_old = model_critic_old(batch_states.type(dtype).detach())
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                total_loss_api.backward()
                loss_value.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                # --------------------------------------------------------------------------------
            # test and save the model
            epoch += 1
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(loss_api.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))

            if epoch % UPDATE_INTERVAL == 0:
                logging.info("Model saved in file")
                # valid(model_actor, epoch, test_log_file, episode_steps)
                valid(model_actor, epoch, test_log_file, val_all_chunks, merged_data_map_val, all_datasets_times)
                ent_coeff = 0.95 * ent_coeff

            if epoch >= 20000:
                break


def calculate_entropy(probs):
    """Calculate the entropy of the policy distribution."""
    log_probs = torch.log(probs + 1e-6)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy


def calculate_prob_ratio(new_probs, old_probs, actions):
    """Calculate the ratio of new and old probabilities for selected actions."""
    new_action_probs = torch.gather(new_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    old_action_probs = torch.gather(old_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    ratio = new_action_probs / (old_action_probs + 1e-6)
    return ratio


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_ppo()


