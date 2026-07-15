import os
import numpy as np
import torch
import torch.optim as optim
from matplotlib.font_manager import weight_dict
from torch.autograd import Variable
import pandas as pd
import logging
from utils import load_trace, C_R, get_seq_chunks_list_by_h5, load_one_trace, get_chunk_data_map, load_h5_file
from PPO import Actor
from IL import ILAgent
from Fusion import FusionActor, FusionCritic
from replay_memory import ReplayMemory
from test import valid_fusion
import env

A_DIM = 100  # 动作维度

FPS = [1.0, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]
RANDOM_SEED = 28

S_RL_INFO = 8
S_IL_INFO = 7
S_LEN = 8

RL_LEARNING_RATE = 1e-3
IL_LEARNING_RATE = 5e-4

SUMMARY_DIR = 'Results_f/'
LOG_FILE = 'Results_f/log'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/ubuntu/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/traces/',118)

# RL = 'Results/RL/RL_20000.model'
# IL= 'Results/IL/IL_20000.model'
RL = 'Results/Fusion/RL_FHVAC_15900.model'
IL= 'Results/Fusion/IL_FHVAC_15900.model'
FUSION = '/home/ubuntu/lyra/FHVAC/Results/Fusion/Fusion_FHVAC_15900.model'
h5_files = ['/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/DETRAC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/DSEC_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/LMOT_desc.h5',
            '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/desc/D²-City_desc.h5',]
# h5_files = ['/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/sort/DETRAC_sort.h5',
#             '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/sort/DSEC_sort.h5',
#             '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/sort/LMOT_sort.h5',
#             '/home/ubuntu/lyra/CASVA/h5_file/sample_cpa/sort/D²-City_sort.h5',]
encoding_files = ['/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DETRAC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_DSEC.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_LMOT.csv',
                  '/mnt/mydisk/lyra/RL_Dataset/coding_time/sample-cpa/coding_time_D²-City.csv']
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
NAME = 'FHVAC'

def train_fusion_model():
    os.makedirs('Results_f', exist_ok=True)
    logging.basicConfig(filename=LOG_FILE + f'_central_{NAME}',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + f'_test_{NAME}', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)

        RLmodel = Actor().type(dtype)
        # RLmodel.train()
        RLmodel.load_state_dict(torch.load(RL, weights_only=True))

        ILmodel = ILAgent().type(dtype)
        # ILmodel.train()
        ILmodel.load_state_dict(torch.load(IL, weights_only=True))

        fusion_actor = FusionActor().type(dtype)
        # fusion_critic = FusionCritic().type(dtype)
        # fusion_actor.train()
        # fusion_critic.train()
        fusion_actor.load_state_dict(torch.load(FUSION, weights_only=True))
        # optimizer_actor = torch.optim.Adam(fusion_actor.parameters(), lr=1e-4)
        # optimizer_critic = torch.optim.Adam(fusion_critic.parameters(), lr=1e-4)
        # # optimizer_rl = optim.Adam(RLmodel.parameters(), lr=RL_LEARNING_RATE)
        # optimizer_il = optim.Adam(ILmodel.parameters(), lr=IL_LEARNING_RATE)
        # optimizer_rl = optim.Adam(RLmodel.parameters(), lr=RL_LEARNING_RATE)


        # 总视频序列
        VIDEO_TOTAL = 27
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
                    # sum_seq = 55
                    # train_num = 40
                    # max_seq = 106
                    sum_seq = 51
                    train_num = 0
                    max_seq = 51

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
        epoch = 0
        batch_size = 256
        gamma = 0.99
        gae_param = 0.97
        clip = 0.2
        ent_coeff = 0.9
        memory = ReplayMemory(1300)
        valid_fusion(RLmodel, ILmodel, fusion_actor, epoch, test_log_file, val_all_chunks, merged_data_map_val,
                     all_datasets_times)

        # while True:
        #     net_env = env.Environment(ALL_BW[0], merged_data_map_train)
        #     state_RL = np.zeros((S_RL_INFO, S_LEN))
        #     state_RL = torch.from_numpy(state_RL)
        #     state_IL = np.zeros((S_IL_INFO, S_LEN))
        #     state_IL = torch.from_numpy(state_IL)
        #     current_video_id = 0
        #     pass_num = 0
        #     for explore in range(exploration_size):
        #         states_RL = []
        #         states_IL = []
        #         actions = []
        #         rewards = []
        #         values = []
        #         returns = []
        #         advantages = []
        #         # net_env = env.Environment(ALL_BW)
        #         # net_env.SEQ_CHUNKS = episode_steps[explore]
        #         # net_env.SEQ_ID = explore
        #
        #         if current_video_id < 3 and explore == dataset_map[current_video_id]:
        #             # pass_num += dataset_pass_num[current_video_id]
        #             current_video_id += 1
        #         if current_video_id == 0 or current_video_id == 3:
        #             net_env.FRAME = FRAMES[0]
        #         else:
        #             net_env.FRAME = FRAMES[1]
        #         video_encoding_time = all_datasets_times[current_video_id]
        #         net_env.SEQ_CHUNKS = train_all_chunks[explore]
        #         net_env.SEQ_ID = explore
        #         net_env.video_chunk_counter = 0
        #
        #         # state_RL = np.zeros((S_RL_INFO, S_LEN))
        #         # state_RL = torch.from_numpy(state_RL)
        #         # state_IL = np.zeros((S_IL_INFO, S_LEN))
        #         # state_IL = torch.from_numpy(state_IL)
        #         last_buff = 0
        #
        #         end_of_video = False
        #
        #         while not end_of_video:
        #             # ------------------------------------------
        #             frl = RLmodel.get_feature(state_RL.unsqueeze(0).type(dtype))
        #             fil = ILmodel.get_feature(state_IL.unsqueeze(0).type(dtype))
        #             prob = fusion_actor(frl, fil)
        #             action = prob.multinomial(num_samples=1).detach()
        #             v = fusion_critic(frl, fil).detach().cpu()
        #             values.append(v)
        #             knob = int(action.squeeze().cpu().numpy())
        #             actions.append(torch.tensor([action]))
        #             states_RL.append(state_RL.unsqueeze(0))
        #             states_IL.append(state_IL.unsqueeze(0))
        #
        #             qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        #             remainder = knob % 20
        #             skip = remainder // 5  # 0 to 3
        #             re = remainder % 5  # 0 to 3
        #             et = video_encoding_time[knob]
        #
        #             _, bw, latency, buffer_size, size, dynamics, f1, Q, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
        #             # reward = 2 * f1 - 1.5 * max((latency - 2), 0) + 0.5 * max(last_buff - buffer_size, 0)
        #             reward = 4 * f1 - 1.5 * max((net_env.lag[-1] - 2), 0)
        #             # reward = 2 * f1 - latency
        #             # reward = C_R(f1, latency)
        #             rewards.append(reward)
        #
        #             last_buff = buffer_size
        #
        #             state_RL = np.roll(state_RL, -1, axis=1)
        #             state_RL[0, -1] = bw
        #             state_RL[1, -1] = latency
        #             state_RL[2, -1] = buffer_size
        #             state_RL[3, -1] = qp
        #             state_RL[4, -1] = skip
        #             state_RL[5, -1] = re
        #             state_RL[6, -1] = size
        #             state_RL[7, -1] = dynamics
        #             state_RL = torch.from_numpy(state_RL)
        #
        #             state_IL = np.roll(state_IL, -1, axis=1)
        #             state_IL[0, -1] = bw
        #             state_IL[1, -1] = latency
        #             state_IL[2, -1] = buffer_size
        #             state_IL[3, -1] = qp
        #             state_IL[4, -1] = skip
        #             state_IL[5, -1] = re
        #             state_IL[6, -1] = Q
        #             state_IL = torch.from_numpy(state_IL)
        #
        #         R = torch.zeros(1, 1)
        #         if not end_of_video:
        #             v = fusion_critic(frl, fil).detach().cpu()
        #             R = v.data
        #         values.append(R)
        #         R = Variable(R)
        #         A = Variable(torch.zeros(1, 1))
        #         for i in reversed(range(len(rewards))):
        #             delta = rewards[i] + gamma * values[i + 1] - values[i]
        #             A = delta + gamma * gae_param * A
        #             advantages.insert(0, A)
        #             R = rewards[i] + gamma * R
        #             returns.insert(0, R)
        #         advantages = torch.stack(advantages)
        #         returns = torch.stack(returns)
        #         memory.push([states_RL, states_IL, actions, returns, advantages])
        #
        #     # update
        #     fusion_actor_old = FusionActor().type(dtype)
        #     fusion_actor_old.load_state_dict(fusion_actor.state_dict())
        #     fusion_critic_old = FusionCritic().type(dtype)
        #     fusion_critic_old.load_state_dict(fusion_critic.state_dict())
        #
        #     avg_policy_loss = 0
        #     avg_value_loss = 0
        #     avg_entropy_loss = 0
        #     avg_ratio = 0
        #     update_count = 0
        #
        #     for flag in range(1):
        #         batch_states_RL, batch_states_IL, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)
        #         frl = RLmodel.get_feature(batch_states_RL.type(dtype))
        #         fil = ILmodel.get_feature(batch_states_IL.type(dtype))
        #         probs_old = fusion_actor_old(frl.type(dtype).detach(), fil.type(dtype).detach())
        #         probs_new = fusion_actor(frl.type(dtype), fil.type(dtype))
        #         ratio = calculate_prob_ratio(probs_new, probs_old, batch_actions)
        #         advantages = batch_advantages.type(dtype)
        #         surr1 = ratio * advantages
        #         surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        #         loss_api = -torch.mean(torch.min(surr1, surr2))
        #         entropy = calculate_entropy(probs_new)
        #         loss_ent = -ent_coeff * entropy
        #         total_loss_api = loss_api + loss_ent
        #
        #         v_pre = fusion_critic(frl.type(dtype), fil.type(dtype))
        #         v_pre_old = fusion_critic_old(frl.type(dtype).detach(), fil.type(dtype).detach())
        #         vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
        #         v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
        #         vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
        #         loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
        #
        #         optimizer_il.zero_grad()
        #         optimizer_rl.zero_grad()
        #         optimizer_actor.zero_grad()
        #         optimizer_critic.zero_grad()
        #         total_loss_api.backward(retain_graph=True)
        #         loss_value.backward()
        #         # total_loss.backward()
        #         optimizer_actor.step()
        #         optimizer_critic.step()
        #         optimizer_rl.step()
        #         optimizer_il.step()
        #
        #         avg_policy_loss += total_loss_api.detach().cpu().numpy()
        #         avg_value_loss += loss_value.detach().cpu().numpy()
        #         avg_entropy_loss += loss_ent.detach().cpu().numpy()
        #         avg_ratio += ratio.mean().detach().cpu().numpy()
        #         update_count += 1
        #     epoch += 1
        #     memory.clear()
        #     # logging.info('Epoch: ' + str(epoch) +
        #     #              ' Avg_policy_loss: ' + str(loss_api.detach().cpu().numpy()) +
        #     #              ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
        #     #              ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))
        #
        #     if update_count > 0:
        #         avg_policy_loss /= update_count
        #         avg_value_loss /= update_count
        #         avg_entropy_loss /= update_count
        #
        #     logging.info('Epoch: ' + str(epoch) +
        #                  ' Avg_policy_loss: ' + str(avg_policy_loss) +
        #                  ' Avg_value_loss: ' + str(avg_value_loss) +
        #                  ' Avg_entropy_loss: ' + str(A_DIM * avg_entropy_loss) +
        #                  ' Avg_ratio: ' + str(avg_ratio))
        #     if epoch % 100 == 0:
        #         logging.info("Model saved in file")
        #         # valid_fusion(RLmodel, ILmodel, fusion_actor, epoch, test_log_file, episode_steps)
        #         valid_fusion(RLmodel, ILmodel, fusion_actor, epoch, test_log_file, val_all_chunks, merged_data_map_val, all_datasets_times)
        #         ent_coeff = 0.95 * ent_coeff
        #         if epoch >= 20000:
        #             break


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_fusion_model()





