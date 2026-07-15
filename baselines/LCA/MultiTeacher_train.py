import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import pandas as pd

from flow_mag_loader import MultiFlowMagLoader
from utils import load_trace, load_one_trace, get_seq_chunks_list_by_h5, get_chunk_data_map
from module import WeightAdapter, ResidualBlockFusion as RBF
from IL import ILAgent
from RL import RLActor
from replay_memory import ReplayMemory
from test import valid_Multi
import env
from config import *
from Student import StudentActor, StudentCritic


# --------------------------------
A_DIM = 100  # 动作维度，qp可选数量 * re可选数量 * fps可选数量 = 5 * 4 * 4 = 80
# --------------------------------
RANDOM_SEED = 28
S_RL_INFO = 6
S_IL_INFO = 6
S_Stu_INFO = 7
S_LEN = 8
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
UPDATE_INTERVAL = 100
L = 2
FPS = 25
# CHUNK_NUM = len(os.listdir("/home/dell/lyra/CASVA/dataset/video_DETRAC_train"))

SUMMARY_DIR = 'Results/'
LOG_FILE = 'Results/log'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/dell/lyra/CASVA/train_trace/')
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)

RL = '/home/dell/lyra/LCA/Results/RL/RL_20000.model'
IL= '/home/dell/lyra/LCA/Results/IL/IL_20000.model'
# h5_file = '/home/dell/lyra/CASVA/h5_file/FHVAC-CASVA/DETRAC_train.h5'
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
NAME = 'Student1'

flow_h5_paths = ["/home/dell/lyra/LCA/flow_mag/flows_DETRAC.h5",
                 "/home/dell/lyra/LCA/flow_mag/flows_DSEC.h5",
                 "/home/dell/lyra/LCA/flow_mag/flows_LMOT.h5",
                 "/home/dell/lyra/LCA/flow_mag/flows_D²-City.h5"]
mag_h5_paths = ["/home/dell/lyra/LCA/flow_mag/mags_DETRAC.h5",
                "/home/dell/lyra/LCA/flow_mag/mags_DSEC.h5",
                "/home/dell/lyra/LCA/flow_mag/mags_LMOT.h5",
                "/home/dell/lyra/LCA/flow_mag/mags_D²-City.h5",]

def compute_logit_loss(student_logits, teacher_logits_list, weights):
    student_probs = F.log_softmax(student_logits / DISTILL_TEMPERATURE, dim=-1)
    teacher_probs = torch.stack([F.softmax(tl / DISTILL_TEMPERATURE, dim=-1) for tl in teacher_logits_list], dim=0)

    weights = weights.float()
    w_sum = weights.sum(dim=1, keepdim=True) + 1e-9
    w_norm = weights / w_sum  # [B, K]

    teacher_probs = teacher_probs.permute(1, 0, 2)
    student_log_exp = student_probs.unsqueeze(1)
    per_teacher_term = (teacher_probs * student_log_exp).sum(dim=-1)  # [B, K]
    per_sample_loss = (w_norm * per_teacher_term).sum(dim=1)  # [B]
    loss = - per_sample_loss.mean()

    return loss

def compute_feat_loss(student_feats,
                             teacher_feats_list,
                             weights: torch.Tensor,
                             normalize_weights=True,
                             eps=1e-9):
    """
    Compute L_fea = mean_b [ sum_k w_{b,k} || teacher_k[b] - student[b] ||_2^2 ].

    Args:
        student_feats: Tensor [B, D]
        teacher_feats_list: list of K tensors, each [B, D]
        weights: Optional tensor [B, K]. If None, use equal weights.
        normalize_weights: whether to normalize weights per-sample so sum_k w=1
        eps: small value to avoid division by zero
    Returns:
        loss: scalar tensor
    """
    device = student_feats.device
    # Stack teacher features -> [B, K, D]
    teachers = torch.stack([t.to(device) for t in teacher_feats_list], dim=1)  # [B, K, D]

    s = student_feats.unsqueeze(1)  # [B, 1, D]
    sq = (teachers - s).pow(2).sum(dim=2)  # [B, K]
    w = weights.to(device).float()

    if normalize_weights:
        w = w / (w.sum(dim=1, keepdim=True) + eps)  # [B, K]

    # per-sample weighted sum over teachers -> [B]
    per_sample = (w * sq).sum(dim=1)
    loss = per_sample.mean()

    return loss

def train_ppo():
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    logging.basicConfig(filename=LOG_FILE + f'/log_central_{NAME}',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + f'/log_test_{NAME}', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)

        RLmodel = RLActor().type(dtype)
        RLmodel.eval()
        RLmodel.load_state_dict(torch.load(RL, weights_only=True))

        ILmodel = ILAgent().type(dtype)
        ILmodel.eval()
        ILmodel.load_state_dict(torch.load(IL, weights_only=True))

        for param in RLmodel.parameters():
            param.requires_grad = False
        for param in ILmodel.parameters():
            param.requires_grad = False

        student_actor = StudentActor().type(dtype)
        student_critic = StudentCritic().type(dtype)
        student_actor.train()
        student_critic.train()
        optimizer_actor = torch.optim.Adam(student_actor.parameters(), lr=PPO_LEARNING_RATE)
        optimizer_critic = torch.optim.Adam(student_critic.parameters(), lr=PPO_LEARNING_RATE)

        weight_adapter = WeightAdapter().type(dtype)
        rbf_rule = RBF(
            early_dim=IL_HIDDEN_SIZE * 4,
            late_dim=IL_HIDDEN_SIZE * 4,
            hidden_dim=PPO_HIDDEN_SIZE
        ).type(dtype)  # 根据实际特征维度调整
        rbf_rl = RBF(
            early_dim=PPO_HIDDEN_SIZE * 5,
            late_dim=PPO_HIDDEN_SIZE,
            hidden_dim=PPO_HIDDEN_SIZE
        ).type(dtype)
        optimizer_weight_adapter = torch.optim.Adam(weight_adapter.parameters(), lr=PPO_LEARNING_RATE)
        optimizer_rbf_rule = torch.optim.Adam(rbf_rule.parameters(), lr=PPO_LEARNING_RATE)
        optimizer_rbf_rl = torch.optim.Adam(rbf_rl.parameters(), lr=PPO_LEARNING_RATE)

        # 总视频序列
        VIDEO_TOTAL = 27
        exploration_size = 0
        # 其实意味着要处理1800个chunk，因为我们这里总共只有480个chunk，所以可以设置为480
        # episode_steps = get_seq_chunks_list(VIDEO_TOTAL) # 1800
        dataset_map = [20, 31, 39]
        dataset_pass_num = [7, 9, 3]
        merged_data_map = {}
        seq_start = 0
        val_all_chunks = []  # 全部验证集的块数列表
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
                exploration_size += test_num

                val_chunks = seq_chunks_list[test_num:]
                test_chunks = seq_chunks_list[:test_num]
                val_all_chunks.extend(val_chunks)
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
        # episode_steps = get_seq_chunks_list_by_h5(h5_file)
        # seq_chunk_data = get_chunk_data_map(h5_file)
        loader = MultiFlowMagLoader(flow_h5_paths, mag_h5_paths, RLmodel)
        # last_knob = 40
        update_num = PPO_UPDATE_NUM
        batch_size = PPO_BATCH_SIZE # 256
        gamma = PPO_GAMMA
        gae_param = PPO_GAE_LAMBDA
        clip = PPO_CLIP_EPSILON
        ent_coeff = PPO_ENTROPY_COEF
        memory = ReplayMemory(exploration_size * 65)

        epoch = 0

        # 其实隐含了训练 30000 epoches
        while True:
            current_video_id = 0
            pass_num = 0
            net_env = env.Environment(ALL_BW[0], merged_data_map)
            for explore in range(exploration_size):
                # net_env = env.Environment(ALL_BW[0], seq_chunk_data)
                # net_env.SEQ_CHUNKS = episode_steps[explore]
                # net_env.SEQ_ID = explore

                if current_video_id < 3 and explore == dataset_map[current_video_id]:
                    pass_num += dataset_pass_num[current_video_id]
                    current_video_id += 1
                if current_video_id == 0 or current_video_id == 3:
                    net_env.FRAME = FRAMES[0]
                else:
                    net_env.FRAME = FRAMES[1]
                video_encoding_time = all_datasets_times[current_video_id]
                net_env.SEQ_CHUNKS = test_all_chunks[explore]
                net_env.SEQ_ID = explore + pass_num
                net_env.video_chunk_counter = 0

                state_RL = np.zeros((S_RL_INFO, S_LEN))
                state_RL = torch.from_numpy(state_RL)
                state_IL = np.zeros((S_IL_INFO, S_LEN))
                state_IL = torch.from_numpy(state_IL)
                state_Stu = np.zeros((S_Stu_INFO, S_LEN))
                state_Stu = torch.from_numpy(state_Stu)

                states_IL = []
                states_RL = []
                states_Stu = []
                flow_feats = []
                actions = []
                rewards = []
                values = []
                returns = []
                advantages = []
                end_of_video = False

                # 每个epoch会执行CHUNK_NUM次actor决策
                while not end_of_video:
                    flow_feat = loader.get_flow_feat(net_env.SEQ_ID, net_env.video_chunk_counter)
                    flow_feat = torch.from_numpy(flow_feat).float()
                    mag = loader.get_mag(net_env.SEQ_ID, net_env.video_chunk_counter)
                    prob, _, _ = student_actor(state_Stu.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    v = student_critic(state_Stu.unsqueeze(0).type(dtype)).detach().cpu()

                    values.append(v)

                    states_IL.append(state_IL.unsqueeze(0))
                    states_RL.append(state_RL.unsqueeze(0))
                    states_Stu.append(state_Stu.unsqueeze(0))
                    flow_feats.append(flow_feat)

                    knob = int(action.squeeze().cpu().numpy())
                    actions.append(torch.tensor([action]))

                    qp = knob // 20  # 0 to 10, because 80 // 16 = 5
                    remainder = knob % 20
                    skip = remainder // 5  # 0 to 3
                    re = remainder % 5  # 0 to 3
                    et = video_encoding_time[knob]

                    bw, latency, size, f1, Q, reward, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
                    rewards.append(torch.tensor([[reward]], dtype=torch.float32))

                    state_Stu = np.roll(state_Stu, -1, axis=1)
                    state_Stu[0, -1] = bw
                    state_Stu[1, -1] = latency
                    state_Stu[2, -1] = size
                    state_Stu[3, -1] = qp
                    state_Stu[4, -1] = skip
                    state_Stu[5, -1] = re
                    state_Stu[6, -1] = mag
                    state_Stu = torch.from_numpy(state_Stu)

                    state_IL = np.roll(state_IL, -1, axis=1)
                    state_IL[0, -1] = bw
                    state_IL[1, -1] = latency
                    state_IL[2, -1] = Q
                    state_IL[3, -1] = qp
                    state_IL[4, -1] = skip
                    state_IL[5, -1] = re
                    state_IL = torch.from_numpy(state_IL)

                    state_RL = np.roll(state_RL, -1, axis=1)
                    state_RL[0, -1] = bw
                    state_RL[1, -1] = latency
                    state_RL[2, -1] = size
                    state_RL[3, -1] = qp
                    state_RL[4, -1] = skip
                    state_RL[5, -1] = re
                    state_RL = torch.from_numpy(state_RL)

                # 如果episode_steps的值小于视频的总块数，那么跳出循环的条件就不是通过end_of_video，此时end_of_video的值就为False
                R = torch.zeros(1, 1)
                if not end_of_video:
                    v = student_critic(state_Stu.unsqueeze(0).type(dtype)).detach().cpu()
                    R = v.data
                # ================================结束一个ep========================================

                # 这里的v.data是从评论家网络输出的值，v.data是一个没有梯度信息的张量
                # Variable(v.data)会将其转换为一个可以进行反向传播的Variable对象
                values.append(R)
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    delta = rewards[i] + gamma * values[i + 1] - values[i]
                    A = delta + gamma * gae_param * A
                    advantages.insert(0, A)
                    R = rewards[i] + gamma * R
                    returns.insert(0, R)
                advantages = torch.stack(advantages)
                returns = torch.stack(returns)
                memory.push([states_Stu, states_IL, states_RL, flow_feats, actions, returns, advantages])

            model_actor_old = StudentActor().type(dtype)
            model_actor_old.load_state_dict(student_actor.state_dict())
            model_critic_old = StudentCritic().type(dtype)
            model_critic_old.load_state_dict(student_critic.state_dict())

            for update_step in range(update_num):
                student_actor.zero_grad()
                student_critic.zero_grad()

                batch_states_stu, batch_states_IL, batch_states_RL, batch_flow_feats, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)

                # --------------------------------------------------------------------------------
                # Calculate policy loss
                probs_old,_,_ = model_actor_old(batch_states_stu.type(dtype).detach())
                probs_new, logits_new, feats_new = student_actor(batch_states_stu.type(dtype))
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
                v_pre = student_critic(batch_states_stu.type(dtype))
                v_pre_old = model_critic_old(batch_states_stu.type(dtype).detach())
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))

                # --------------------------------------------------------------------------------
                # Calculate logits and feats loss
                with torch.no_grad():
                    early_IL, late_IL, logit_IL = ILmodel.get_feature(batch_states_IL.type(dtype))
                    early_RL, late_RL, logit_RL = RLmodel.get_feature(batch_states_RL.type(dtype), batch_flow_feats.type(dtype))

                teacher_logits = [logit_IL, logit_RL]
                weights = weight_adapter(teacher_logits)
                logit_loss = compute_logit_loss(logits_new, teacher_logits, weights)

                fused_feat_IL = rbf_rule(early_IL, late_IL)
                fused_feat_RL = rbf_rl(early_RL, late_RL)
                teacher_feats_list = [fused_feat_IL, fused_feat_RL]
                feat_loss = compute_feat_loss(feats_new, teacher_feats_list, weights)

                total_loss = total_loss_api + logit_loss + feat_loss

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                optimizer_weight_adapter.zero_grad()
                optimizer_rbf_rule.zero_grad()
                optimizer_rbf_rl.zero_grad()
                total_loss.backward()
                loss_value.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                optimizer_weight_adapter.step()
                optimizer_rbf_rule.step()
                optimizer_rbf_rl.step()
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
                valid_Multi(student_actor, epoch, test_log_file, val_all_chunks, merged_data_map, loader, all_datasets_times)
                ent_coeff = 0.95 * ent_coeff
            if epoch >= 50000:
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


