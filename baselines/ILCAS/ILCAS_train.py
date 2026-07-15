import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
import math
import json
import env
from Expert import ExpertDP
from ILCAS import DiscriminatorNet, Actor, Critic
from replay_memory import ReplayMemory
from test import valid
from utils import load_trace, get_seq_chunks_list_by_h5, load_one_trace, get_chunk_data_map, load_h5_file
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy

import warnings
import time
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.torch_inductor_cache")
torch._dynamo.config.cache_size_limit = 128


# --------------------------------
A_DIM = 100  # 动作维度，qp可选数量 * re可选数量 * fps可选数量 = 6 * 5 * 4 = 120
# --------------------------------
RANDOM_SEED = 28
S_INFO = 8  # chunk_volume, bw, latency, qp, skip, r, buffer, motion_feature
S_LEN = 8  # past 8
UPDATE_INTERVAL = 100
T_CHUNK      = 2.0       # 每块持续时间 (秒)
LAG_MAX      = 2.0       # Expert 最大可接受上传延迟 (论文设 L=1s)
LR_ACTOR     = 5e-4      # PPO 策略网络学习率
LR_DISC      = 1e-4      # 判别器学习率
# 720p
# RE = [[1280, 720], [960, 540], [854, 480], [426, 240]]
# LMOT
# RE = [[1800, 1000], [1296, 720], [1080, 600], [864, 480]]
# DETRAC
RE = [[960, 540], [854, 480], [640, 360], [426, 240]]
# DSEC
# RE = [[1440, 1080], [1080, 810], [960, 720], [720, 540], [480, 360]]

H4, W4 = 36, 36
# CHUNK_NUM = len(os.listdir("/home/dell/lyra/CASVA/dataset/video_DETRAC_train"))

SUMMARY_DIR = 'Results/'
LOG_FILE = 'Results/log'
NAME = 'ILCAS2'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('/home/dell/lyra/CASVA/train_trace/')
LEAF_SPLITS_PATH = '/home/dell/lyra/Sec3/cluster_method/results/meanshift/2.828-0.13/meanshift_leaf_splits.json'
# ALL_BW, ALL_NAME = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)

h5_files = ['/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DETRAC_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DSEC_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/LMOT_desc.h5',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/D²-City_desc.h5',]
csv_files = ['/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc_info/DETRAC_desc.csv',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc_info/DSEC_desc.csv',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc_info/LMOT_desc.csv',
            '/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc_info/D²-City_desc.csv',]
encoding_files = ['/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_DETRAC.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_DSEC.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_LMOT.csv',
                  '/home/dell/lyra/ILCAS/coding_time/sample-cpa/coding_time_D²-City.csv']
feature_map_dir = ['/home/dell/lyra/ILCAS/mv_chunks/DETRAC',
                   '/home/dell/lyra/ILCAS/mv_chunks/DSEC',
                   '/home/dell/lyra/ILCAS/mv_chunks/LMOT',
                   '/home/dell/lyra/ILCAS/mv_chunks/D²-City',]
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}

DS_SPLIT_INFO = {
    #             train_start  val_start  start_idx  train_num
    'DETRAC':  (  0,           0,         0,         10),
    'DSEC':    ( 10,           4,         0,          5),
    'LMOT':    ( 15,           7,         0,          6),
    'D²-City': ( 21,          10,        51,         40),
}
def load_leaf_whitelists(json_path, leaf_id):
    """
    返回 (train_seq_set, val_seq_set)
    只记录全局视频编号，不到块级别。
    leaf_id=None 时均返回 None。
    """
    if leaf_id is None:
        return None, None
    with open(json_path) as f:
        splits = json.load(f)
    key = str(leaf_id)
    train_wl, val_wl = set(), set()
    seen = set()   # 去重，同一视频多个块只处理一次
    for s in splits[key]['samples']:
        vid = s['video_id']
        ds  = s['ds_name']
        if (ds, vid) in seen:
            continue
        seen.add((ds, vid))
        train_start, val_start, start_idx, train_num = DS_SPLIT_INFO[ds]
        if vid < start_idx + train_num:
            train_wl.add(train_start + (vid - start_idx))
        else:
            val_wl.add(val_start + (vid - (start_idx + train_num)))
    print(f"[Leaf {leaf_id}] 训练白名单: {len(train_wl)} 条视频  验证白名单: {len(val_wl)} 条视频")
    return train_wl, val_wl

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

def get_acc_ul_tb(data_map, seq_id, chunk_nums, cooked_trace):
    acc = np.zeros((chunk_nums, A_DIM))
    ul = np.zeros((chunk_nums, A_DIM))
    for i in range(chunk_nums):
        for j in range(A_DIM):
            # qp = j // 20  # 0 to 10, because 80 // 16 = 5
            # remainder = j % 20
            # s = remainder // 5  # 0 to 3
            # r = remainder % 5  # 0 to 3
            # acc[i, j] = df.loc[(seq_id, i, qp, s, r), 'Accuracy']
            # chunk_size = df.loc[(seq_id, i, qp, s, r), 'Size']
            chunk_size, chunk_acc, _ = data_map[(seq_id, i, j)]
            acc[i, j] = chunk_acc
            ul[i, j] = cal_upload_lag(cooked_trace, chunk_size)
    return acc,ul

def fast_gae_to_list(rewards, values, gamma=0.95, gae_param=0.97, bootstrap=0.0):
    """
    rewards : list[float]                        # len = T
    values  : list[Tensor] 每个 shape=(1,1)      # len = T
    returns / advantages -> list[Tensor] 同样 (1,1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(rewards, list):
        r = torch.tensor(rewards, device=device, dtype=torch.float32)      # (T,)
        v = torch.cat([v.view(-1) for v in values]).to(device)             # (T,)
        tensor_input = False
    elif isinstance(rewards, torch.Tensor):
        r = rewards.detach().float().view(-1).to(device)
        v = values.detach().float().view(-1).to(device)
        tensor_input = True
    else:
        raise TypeError(f"Unsupported rewards type: {type(rewards)}")

    v_next = torch.cat([v[1:], torch.tensor([bootstrap], device=device)])

    # -------- 2. GPU 上反向递推 GAE ----------
    deltas = r + gamma * v_next - v                         # δ_t
    T = deltas.size(0)
    adv = torch.zeros(T, device=device)
    A = 0.0
    for t in range(T - 1, -1, -1):
        A = deltas[t] + gamma * gae_param * A
        adv[t] = A
    ret = adv + v                                           # R_t

    if tensor_input:
        return ret.view(-1, 1), adv.view(-1, 1)
    else:
        returns_list = [ret[i].view(1, 1) for i in range(T)]
        advantages_list = [adv[i].view(1, 1) for i in range(T)]
        return returns_list, advantages_list


def fast_return_to_list(rewards, gamma=0.95):
    """
    基于 ILCAS 论文的无 critic 模式:
    直接用累积折扣回报作为 advantage（即 A_t = G_t）

    rewards : list[float]             # discriminator 输出 log(D(s,a))
    gamma   : float                   # 折扣系数, 论文设定为 0.95
    return:
        returns_list, advantages_list
        每个元素 shape = (1,1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(rewards, list):
        r = torch.tensor(rewards, dtype=torch.float32, device=device)
        tensor_input = False
    elif isinstance(rewards, torch.Tensor):
        r = rewards.detach().float().view(-1).to(device)
        tensor_input = True
    else:
        raise TypeError(f"Unsupported rewards type: {type(rewards)}")

    T = len(r)
    G = torch.zeros(T, device=device)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = r[t] + gamma * running_return
        G[t] = running_return

    # 在无 critic 的情况下，直接用 G_t 作为 advantage
    advantages = G.clone()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if tensor_input:
        return advantages.view(-1, 1)
    else:
        advantages_list = [advantages[i].view(1, 1) for i in range(T)]
        return advantages_list

@torch.no_grad()
def extract_motion_features(video_motion_map, model_mfcnn, video_size, device='cuda', batch_size=64):
    """
    从 video_motion_map 批量提取 motion 特征
    Args:
        video_motion_map: List[List[np.ndarray]]   # 每个元素是 (36,36) float32, [0,1]
        model_mfcnn: MotionFeatureCNN 实例
        video_size: video nums
        device: 'cuda' or 'cpu'
        batch_size: 每次提取的块数
    Returns:
        video_motion_feat: List[List[Tensor]]  # 每个特征 shape = (motion_dim,)
    """
    model_mfcnn.eval()
    model_mfcnn.to(device)
    video_motion_feat = []

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        for seq_idx, chunk_list in enumerate(video_motion_map):
            if seq_idx >= video_size:
                break
            seq_feats = []

            # stack 成一个 tensor
            tensor_chunks = torch.stack([
                torch.from_numpy(chunk).unsqueeze(0)  # (36,36)
                for chunk in chunk_list
            ])  # (N,36,36)

            dataset = TensorDataset(tensor_chunks)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            for (batch,) in tqdm(loader, leave=False, desc=f"Seq {seq_idx}"):
                batch = batch.to(device, non_blocking=True)
                feats = model_mfcnn(batch)            # (B, motion_dim)
                seq_feats.append(feats.cpu())         # 移回 CPU 节省显存

            seq_feats = torch.cat(seq_feats, dim=0)   # (num_chunks, motion_dim)
            video_motion_feat.append(seq_feats)

    return video_motion_feat


def adaptive_update_schedule(d_expert, d_agent):
    """根据判别器性能动态调整更新频率"""
    gap = d_expert - d_agent

    if gap < 0.05:  # 判别器太弱，无法区分
        return 3, 1  # (disc_updates, actor_updates)
    elif gap > 0.3:  # 判别器太强
        return 1, 5
    else:  # 平衡状态
        return 2, 3

def train_ilcas():
    os.makedirs('Results', exist_ok=True)
    logging.basicConfig(filename=LOG_FILE + f'_central_{NAME}',
                        filemode='w',
                        level=logging.INFO)
    # ★ 新增：加载白名单
    train_wl, val_wl = load_leaf_whitelists(LEAF_SPLITS_PATH, 6)
    with open(LOG_FILE + f'_test_{NAME}', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
        # 总视频序列
        VIDEO_TOTAL = 27 # 27

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
        expert = ExpertDP()
        video_actions = []
        train_motion_map = []
        val_motion_map = []
        for h5_path, csv_path, fm_dir in zip(h5_files, csv_files, feature_map_dir):
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
                print(f"train_{dataset_name}: seq范围 [{seq_train_start}, {seq_train_start + train_num - 1}]，共 {len(data_map)} 条。")
                seq_train_start += train_num

                data_map = get_chunk_data_map(tmp_df,
                                             seq_min=start_idx + train_num,
                                             seq_max=max_seq - 1,
                                             seq_start=seq_val_start)
                merged_data_map_val.update(data_map)
                print(f"val_{dataset_name}: chunks num {val_all_chunks}")
                print(f"val_{dataset_name}: seq范围 [{seq_val_start}, {seq_val_start + (sum_seq - train_num - 1)}]，共 {len(data_map)} 条。")
                seq_val_start += (sum_seq - train_num)

                # --- 1) 离线生成所有序列的专家示例 ---
                csv_df = pd.read_csv(csv_path)
                seq_list = csv_df["SEQ"].tolist()
                data_map = get_chunk_data_map(tmp_df,
                                              seq_min=start_idx,
                                              seq_max=start_idx + train_num - 1,
                                              seq_start=0)
                for seq in range(sum_seq):
                    chunk_nums = seq_chunks_list[start_idx + seq]
                    chunk_motion_map = []
                    for chunk_id in range(chunk_nums):
                        # old_seq = seq_list[seq]
                        old_seq = seq_list[start_idx + seq]
                        feature_map = env.get_feature_map(old_seq, chunk_id, 0, fm_dir)
                        chunk_motion_map.append(feature_map)
                    if seq < train_num:
                        acc, ul = get_acc_ul_tb(data_map, seq, chunk_nums, ALL_BW[0])
                        video_actions.append(expert.solve(acc, ul))
                        train_motion_map.append(chunk_motion_map)
                    else:
                        val_motion_map.append(chunk_motion_map)
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
        disc_update_num = 1
        actor_update_num = 2
        batch_size = 128  # 256
        gamma = 0.95
        gae_param = 0.97
        clip = 0.2
        ent_coeff = 0.02
        memory_agent = ReplayMemory(1300)
        memory_expert = ReplayMemory(1300)

        epoch = 0

        # --- 2) 初始化模型 ---
        # model_actor = Actor().type(dtype)
        # model_critic = Critic().type(dtype)
        # model_disc = DiscriminatorNet().type(dtype)
        model_actor = Actor().to(device)
        model_actor.load_state_dict(torch.load("/home/dell/lyra/ILCAS/Results/history/ILCAS_20000_0.685968_3.695348.model", weights_only=True))
        model_disc = DiscriminatorNet().to(device)

        # motion_feat_cache_actor = extract_motion_features(video_motion_map, model_actor.mfcnn, exploration_size,device='cuda', batch_size=64)
        # motion_feat_cache_disc = extract_motion_features(video_motion_map, model_disc.mfcnn, exploration_size, device='cuda', batch_size=64)

        # model_actor.train()
        # model_disc.train()

        # optimizer_actor = optim.Adam(model_actor.parameters(), lr=LR_ACTOR)
        # optimizer_disc = optim.Adam(model_disc.parameters(), lr=LR_DISC)

        valid(model_actor, epoch, test_log_file, val_all_chunks, merged_data_map_val, all_datasets_times,
              val_motion_map, ALL_BW[0], ALL_NAME[0], val_wl)

        # # 其实隐含了训练 50000 epoches
        # while True:
        #     # # ---- ⏱ 启动性能分析 ----
        #     # profiler = cProfile.Profile()
        #     # profiler.enable()
        #     # # ----------------------
        #     model_actor_old = Actor().to(device)
        #     model_actor_old.load_state_dict(model_actor.state_dict())
        #     model_actor_old.eval()
        #     # model_critic_old = Critic().to(device)
        #     # model_critic_old.load_state_dict(model_critic.state_dict())
        #     # model_critic_old.eval()
        #     model_disc.eval()
        #
        #     net_env = env.Environment(ALL_BW[0], merged_data_map_train)
        #     net_env_exp = env.Environment(ALL_BW[0], merged_data_map_train)
        #     state = np.zeros((S_INFO - 1, S_LEN))
        #     state = torch.from_numpy(state)
        #     exp_state = np.zeros((S_INFO - 1, S_LEN))
        #     exp_state = torch.from_numpy(exp_state)
        #     current_video_id = 0
        #     pass_num = 0
        #     for explore in range(exploration_size):
        #         # net_env = env.Environment(ALL_BW[0])
        #         # state = np.zeros((S_INFO - 1, S_LEN))
        #         # state = torch.from_numpy(state)
        #         # 1) 收集 Agent 轨迹
        #         if current_video_id < 3 and explore == dataset_map[current_video_id]:
        #             # pass_num += dataset_pass_num[current_video_id]
        #             current_video_id += 1
        #         if train_wl is not None and explore not in train_wl:
        #             continue
        #         if current_video_id == 0 or current_video_id == 3:
        #             net_env.FRAME = FRAMES[0]
        #             net_env_exp.FRAME = FRAMES[0]
        #         else:
        #             net_env.FRAME = FRAMES[1]
        #             net_env_exp.FRAME = FRAMES[1]
        #         video_encoding_time = all_datasets_times[current_video_id]
        #         net_env.SEQ_CHUNKS = train_all_chunks[explore]
        #         net_env.SEQ_ID = explore
        #         net_env.video_chunk_counter = 0
        #         net_env_exp.SEQ_CHUNKS = train_all_chunks[explore]
        #         net_env_exp.SEQ_ID = explore
        #         net_env_exp.video_chunk_counter = 0
        #
        #         # motion_map = np.zeros((H4, W4), np.float32)
        #         # motion_map = torch.from_numpy(motion_map)
        #
        #         states = []
        #         exp_states = []
        #         actions = []
        #         # actor_motion_feats = []
        #         # disc_motion_feats = []
        #         motion_maps = []
        #         # exp_motion_maps = []
        #         # values = []
        #         rewards = []
        #         # returns = []
        #         # advantages = []
        #         end_of_video = False
        #
        #         exp_chunk_actions = video_actions[explore]
        #         expert_actions = [torch.tensor([x]) for x in exp_chunk_actions]
        #         # actor_chunk_feat = motion_feat_cache_actor[explore]
        #         # disc_chunk_feat = motion_feat_cache_disc[explore]
        #         chunk_motion_map = train_motion_map[net_env.SEQ_ID]
        #
        #         # 每个epoch会执行CHUNK_NUM次actor决策
        #         while not end_of_video:
        #             motion_map = torch.from_numpy(chunk_motion_map[net_env.video_chunk_counter]).float()
        #             # torch.compiler.cudagraph_mark_step_begin()
        #             with torch.no_grad():
        #                 print(state.unsqueeze(0).shape, motion_map.unsqueeze(0).shape)
        #                 prob = model_actor_old(state.unsqueeze(0).float().to(device), motion_map.unsqueeze(0).float().to(device))
        #                 action = prob.multinomial(num_samples=1)
        #                 # v = model_critic_old(state.unsqueeze(0).float().to(device), motion_map.unsqueeze(0).float().to(device))
        #                 d_out = model_disc(state.unsqueeze(0).float().to(device), motion_map.unsqueeze(0).float().to(device), action.squeeze(1).long())
        #                 # reward = torch.log(d_out + 1e-8).item()
        #                 reward = torch.log(d_out + 1e-8)
        #
        #             # values.append(v.cpu())
        #             rewards.append(reward.cpu())
        #             actions.append(torch.tensor([action]))
        #             states.append(state.unsqueeze(0))
        #             exp_states.append(exp_state.unsqueeze(0))
        #             motion_maps.append(motion_map.unsqueeze(0))
        #             # actor_motion_feats.append(actor_motion_feat.unsqueeze(0))
        #             # disc_motion_feats.append(disc_motion_feat.unsqueeze(0))
        #
        #             # knob = int(action.item())
        #             knob = int(action.squeeze().cpu().numpy())
        #             qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        #             remainder = knob % 20
        #             skip = remainder // 5  # 0 to 3
        #             re = remainder % 5  # 0 to 3
        #             et = video_encoding_time[knob]
        #
        #             bw, reward_f, latency, buffer_size, size, f1, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
        #             # motion_map = torch.from_numpy(feature_map)
        #
        #             # reward += 2 * reward_f
        #             # rewards.append(torch.tensor([[reward]], dtype=torch.float32))
        #
        #             # state = torch.roll(state, shifts=-1, dims=1)
        #             state = np.roll(state, -1, axis=1)
        #             state[0, -1] = size
        #             state[1, -1] = bw
        #             state[2, -1] = latency
        #             state[3, -1] = qp
        #             state[4, -1] = skip
        #             state[5, -1] = re
        #             state[6, -1] = buffer_size
        #             state = torch.from_numpy(state)
        #
        #             knob = int(exp_chunk_actions[net_env_exp.video_chunk_counter])
        #             # knob = int(action.squeeze().cpu().numpy())
        #             # print(knob)
        #             qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        #             remainder = knob % 20
        #             skip = remainder // 5  # 0 to 3
        #             re = remainder % 5  # 0 to 3
        #             et = video_encoding_time[knob]
        #
        #             bw, reward_f, latency, buffer_size, size, f1, end_of_video_exp = net_env_exp.get_video_chunk(qp, skip, re, et)
        #             # motion_map = torch.from_numpy(feature_map)
        #
        #             exp_state = np.roll(exp_state, -1, axis=1)
        #             exp_state[0, -1] = size
        #             exp_state[1, -1] = bw
        #             exp_state[2, -1] = latency
        #             exp_state[3, -1] = qp
        #             exp_state[4, -1] = skip
        #             exp_state[5, -1] = re
        #             exp_state[6, -1] = buffer_size
        #             exp_state = torch.from_numpy(exp_state)
        #         # ================================结束一个ep========================================
        #
        #         # returns, advantages = fast_gae_to_list(
        #         #     rewards, values,
        #         #     gamma=gamma, gae_param=gae_param,
        #         #     bootstrap=0.0)
        #         # advantages = fast_return_to_list(rewards, gamma=gamma)
        #         # memory_agent.push([states, actions, expert_actions, motion_maps, returns, advantages])
        #         # memory_expert.push([states, expert_actions, motion_maps])
        #         memory_agent.push([states, actions, motion_maps, rewards])
        #         memory_expert.push([exp_states, expert_actions, motion_maps])
        #
        #     model_disc.train()
        #     for update_step in range(disc_update_num):
        #         s_a, a_a, mm_a, _ = memory_agent.sample(batch_size)
        #         s_e, a_e, mm_e = memory_expert.sample(batch_size)
        #         # 统一迁移到 device
        #         s_a, a_a, mm_a = s_a.float().to(device), a_a.long().to(device), mm_a.float().to(device)
        #         s_e, a_e, mm_e = s_e.float().to(device), a_e.long().to(device), mm_e.float().to(device)
        #
        #         # ------- (a) 判别器 BCE -------
        #         out_e = model_disc(s_e, mm_e, a_e)
        #         out_a = model_disc(s_a, mm_a, a_a)
        #         # print(f"D(expert) mean={out_e.mean().item():.4f}, D(agent) mean={out_a.mean().item():.4f}")
        #         loss_d = -(torch.log(out_e + 1e-8).mean() + torch.log(1 - out_a + 1e-8).mean())
        #         optimizer_disc.zero_grad()
        #         loss_d.backward()
        #         optimizer_disc.step()
        #
        #     # model_actor_old = Actor().type(dtype)
        #     # model_actor_old.load_state_dict(model_actor.state_dict())
        #     # model_critic_old = Critic().type(dtype)
        #     # model_critic_old.load_state_dict(model_critic.state_dict())
        #
        #     for update_step in range(actor_update_num):
        #
        #         s_a, a_a, mm_a, rewards, = memory_agent.sample(batch_size, True)
        #         # 统一迁移到 device
        #         s_a, a_a, mm_a, reward = s_a.float().to(device), a_a.long().to(device), mm_a.float().to(device), rewards.float().to(device)
        #
        #         adv_a = fast_return_to_list(rewards, gamma=gamma)
        #         # r_a, adv_a = fast_gae_to_list(rewards, values, gamma=gamma, gae_param=gae_param, bootstrap=0.0)
        #
        #         # Calculate policy loss
        #         # probs_old = model_actor_old(s_a.type(dtype).detach(), m_a.type(dtype).detach())
        #         # probs_new = model_actor(s_a.type(dtype), m_a.type(dtype))
        #         with torch.no_grad():
        #             probs_old = model_actor_old(s_a, mm_a)
        #         probs_new = model_actor(s_a, mm_a)
        #         # ------- 再计算 log D(s,a) 奖励 -------
        #         with torch.no_grad():
        #             out = model_disc(s_a, mm_a, a_a)
        #             # rewards = torch.log(out + 1e-8)
        #         # print(f"D(reward) mean={out.mean().item():.4f}")
        #         # probs_old = model_actor_old(s_a.detach(), mm_a.detach())
        #         # probs_new = model_actor(s_a, mm_a)
        #         ratio = calculate_prob_ratio(probs_new, probs_old, a_a)
        #
        #         # advantages = adv_a.type(dtype)
        #         advantages = adv_a.float()
        #         surr1 = ratio * advantages
        #         surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        #         loss_api = -torch.mean(torch.min(surr1, surr2))
        #         entropy = calculate_entropy(probs_new)
        #         loss_ent = -ent_coeff * entropy
        #         total_loss_api = loss_api + loss_ent
        #
        #         # # Update critic networks
        #         # # v_pre = model_critic(s_a.type(dtype), m_a.type(dtype))
        #         # # v_pre_old = model_critic_old(s_a.type(dtype).detach(), m_a.type(dtype).detach())
        #         # # v_pre = model_critic(s_a, mm_a)
        #         # # v_pre_old = model_critic_old(s_a.detach(), mm_a.detach())
        #         # v_pre = model_critic(s_a, mm_a)
        #         # with torch.no_grad():
        #         #     v_pre_old = model_critic_old(s_a, mm_a)
        #         # # vfloss1 = (v_pre - r_a.type(dtype)) ** 2
        #         # vfloss1 = (v_pre - r_a.float()) ** 2
        #         # v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
        #         # # vfloss2 = (v_pred_clipped - r_a.type(dtype)) ** 2
        #         # vfloss2 = (v_pred_clipped - r_a.float()) ** 2
        #         # loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
        #
        #         optimizer_actor.zero_grad()
        #         # optimizer_critic.zero_grad()
        #         total_loss_api.backward()
        #         # loss_value.backward()
        #         # torch.nn.utils.clip_grad_norm_(model_actor.parameters(), 0.5)
        #         # torch.nn.utils.clip_grad_norm_(model_critic.parameters(), 0.5)
        #         optimizer_actor.step()
        #         # optimizer_critic.step()
        #
        #
        #     # gap = out_e.mean().item() - out_a.mean().item()
        #     # if gap > 0.5:  # 判别器明显强
        #     #     disc_update_num = 1
        #     #     actor_update_num = 3
        #     # elif gap < 0.2:  # 判别器明显弱
        #     #     disc_update_num = 3
        #     #     actor_update_num = 1
        #     # else:  # 平衡区
        #     #     disc_update_num = 2
        #     #     actor_update_num = 2
        #     epoch += 1
        #     memory_agent.clear()
        #     memory_expert.clear()
        #
        #     if epoch % UPDATE_INTERVAL == 0:
        #         logging.info("Model saved in file")
        #         # motion_feat_cache_actor = extract_motion_features(video_motion_map, model_actor.mfcnn, VIDEO_TOTAL, device='cuda', batch_size=64)
        #         valid(model_actor, epoch, test_log_file, val_all_chunks, merged_data_map_val, all_datasets_times, val_motion_map, ALL_BW[0], ALL_NAME[0], val_wl)
        #     if epoch >= 20000:
        #         break


def calculate_entropy(probs):
    """Calculate the entropy of the policy distribution."""
    log_probs = torch.log(probs + 1e-6)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy


def calculate_prob_ratio(new_probs, old_probs, actions):
    """Calculate the ratio of new and old probabilities for selected actions."""
    # new_action_probs = torch.gather(new_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    # old_action_probs = torch.gather(old_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    new_action_probs = torch.gather(new_probs, dim=1, index=actions.unsqueeze(1).long().to(new_probs.device))
    old_action_probs = torch.gather(old_probs, dim=1, index=actions.unsqueeze(1).long().to(new_probs.device))
    ratio = new_action_probs / (old_action_probs + 1e-6)
    return ratio


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_ilcas()


