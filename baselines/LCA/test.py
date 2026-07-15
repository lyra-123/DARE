import time
from collections import deque
import numpy as np
import os

from h5py.tests.test_file_alignment import dataset_name
from tqdm import tqdm
import torch
import torch.nn.functional as F
from RL import RLActor
from IL import ILAgent
from Student import StudentActor
import env
from utils import load_one_trace, count_accuracy
import scipy.stats as stats
import time

# QP = [23, 28, 33, 38, 43]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
SKIP = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]

S_RL_INFO = 6
S_IL_INFO = 6
S_Stu_INFO = 7
S_LEN = 8  # past 8
L = 2
FPS = 25
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}

# RL = 'Results/RL_1600.model'
# IL = 'Results/IL/IL_4000.model'

NAME = 'LCA'
SUMMARY_DIR = 'Results'
LOG_FILE_VALID = 'Results/test_results/log_valid'
TEST_LOG_FOLDER_VALID = 'Results/test_results/'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def evaluation_RL(model, net_env, loader, video_encoding_time, state):
    # state = np.zeros((S_RL_INFO, S_LEN))
    # state = torch.from_numpy(state)
    # time_stamp = 0
    while True:
        flow_feat = loader.get_flow_feat(net_env.SEQ_ID, net_env.video_chunk_counter)
        flow_feat = torch.from_numpy(flow_feat).float()
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype), flow_feat.type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        bw, latency, size, f1, _, reward, end_of_video = net_env.get_video_chunk(qp, skip, re, et)

        state = np.roll(state, -1, axis=1)
        state[0, -1] = bw
        state[1, -1] = latency
        state[2, -1] = size
        state[3, -1] = qp
        state[4, -1] = skip
        state[5, -1] = re
        state = torch.from_numpy(state)

        if end_of_video:
            f1_mean = np.mean(net_env.F1)
            f1_std = np.std(net_env.F1)
            lag_mean = np.mean(net_env.lag)
            lag_std = np.std(net_env.lag)
            rw_mean = np.mean(net_env.Reward)
            rw_std = np.std(net_env.Reward)
            return f1_mean, f1_std, lag_mean, lag_std, rw_mean, rw_std, state


def valid_RL(shared_model, epoch, log_file, val_all_chunks, merged_data_map, loader, all_datasets_times):
    # 检查文件夹是否存在，如果存在则删除
    if os.path.exists(TEST_LOG_FOLDER_VALID):
        os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = RLActor().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',2)
    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    SEQ_TOTAL = 46  # 86400
    SEQ_START = 20
    dataset_map = [7, 16, 19]
    dataset_pass_num = [11, 8, 73]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    env_fix = env.Environment(cooked_trace=cooked_bw, seq_chunk_data=merged_data_map)
    state = np.zeros((S_RL_INFO, S_LEN))
    state = torch.from_numpy(state)
    env_fix.start = 2
    env_fix.video_start_shoot = 0
    current_video_id = 0
    f1_avg, lag_avg, reward_avg = 0., 0., 0.
    A = []
    L = []
    R = []
    for seq_id in range(SEQ_TOTAL):
        # env_fix = env.Environment(cooked_trace=cooked_bw, seq_chunk_data=seq_chunk_data)
        # env_fix.start = 36
        # env_fix.video_start_shoot = 36 - 2
        # env_fix.SEQ_CHUNKS = seq_chunks[SEQ_START+seq_id]
        # env_fix.SEQ_ID = SEQ_START+seq_id
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
        if current_video_id == 0 or current_video_id == 3:
            env_fix.FRAME = FRAMES[0]
        else:
            env_fix.FRAME = FRAMES[1]
        video_encoding_time = all_datasets_times[current_video_id]
        env_fix.SEQ_CHUNKS = val_all_chunks[seq_id]
        env_fix.SEQ_ID = SEQ_START + seq_id
        env_fix.video_chunk_counter = 0
        f1, f1_std, lag, lag_std, rw, rw_std, state = evaluation_RL(model, env_fix, loader, video_encoding_time, state)
        # f1_sum += f1
        # lag_sum += lag
        # reward_sum += rw
        # A.append(f1)
        # L.append(lag)
        # R.append(rw)

        f1_avg = f1
        lag_avg = lag
        reward_avg = rw

    # acc_mean = np.mean(A)
    # lag_mean = np.mean(L)
    # rewards_mean = np.mean(R)
    # print(epoch, acc_mean, lag_mean, rewards_mean)
    # log_file.write(str(int(epoch)) + '\t' +
    #                str(acc_mean) + '\t' +
    #                str(lag_mean) + '\t' +
    #                str(rewards_mean) + '\n')
    # f1_avg = f1_sum / SEQ_TOTAL
    # lag_avg = lag_sum / SEQ_TOTAL
    # reward_avg = reward_sum / SEQ_TOTAL
    print(epoch, cooked_name, f1_avg, lag_avg, reward_avg)
    log_file.flush()
    add_str = 'RL'
    output_path  = SUMMARY_DIR +'/%s' % add_str
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_save_path = output_path + "/%s_%d.model" % (add_str, int(epoch))
    torch.save(shared_model.state_dict(), model_save_path)

def evaluation_Multi(model, net_env, loader, video_encoding_time, state_Stu):
    # state_Stu = np.zeros((S_Stu_INFO, S_LEN))
    # state_Stu = torch.from_numpy(state_Stu)

    end_of_video = False
    while not end_of_video:
        mag = loader.get_mag(net_env.SEQ_ID, net_env.video_chunk_counter)
        with torch.no_grad():
            prob, _, _ = model(state_Stu.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        bw, latency, size, _, _, _, end_of_video = net_env.get_video_chunk(qp, skip, re, et)

        state_Stu = np.roll(state_Stu, -1, axis=1)
        state_Stu[0, -1] = bw
        state_Stu[1, -1] = latency
        state_Stu[2, -1] = size
        state_Stu[3, -1] = qp
        state_Stu[4, -1] = skip
        state_Stu[5, -1] = re
        state_Stu[6, -1] = mag
        state_Stu = torch.from_numpy(state_Stu)

    f1_mean = np.mean(net_env.F1)
    lag_mean = np.mean(net_env.lag)
    Reward_mean = np.mean(net_env.Reward)
    return f1_mean, lag_mean, Reward_mean, state_Stu


def valid_Multi(shared_model, epoch, log_file, val_all_chunks, merged_data_map, loader, all_datasets_times):
    model = StudentActor().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/traces/', 118)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/', 2)

    SEQ_TOTAL = 46  # 86400
    SEQ_START = 20
    dataset_map = [7, 16, 19]
    dataset_pass_num = [11, 8, 73]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    # env_fix = env.Environment(cooked_trace=cooked_bw, seq_chunk_data=merged_data_map)
    state = np.zeros((S_Stu_INFO, S_LEN))
    state = torch.from_numpy(state)
    # env_fix.start = 2
    # env_fix.video_start_shoot = 0
    start = 2
    current_video_id = 0
    f1_avg, lag_avg, reward_avg = 0., 0., 0.

    for seq_id in range(SEQ_TOTAL):
        env_fix = env.Environment(cooked_trace=cooked_bw, seq_chunk_data=merged_data_map)
        # env_fix.start = 36
        # env_fix.video_start_shoot = 36 - 2
        # env_fix.SEQ_CHUNKS = seq_chunks[SEQ_START + seq_id]
        # env_fix.SEQ_ID = SEQ_START + seq_id
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
            start = 2
        if current_video_id == 0 or current_video_id == 3:
            env_fix.FRAME = FRAMES[0]
        else:
            env_fix.FRAME = FRAMES[1]
        video_encoding_time = all_datasets_times[current_video_id]
        env_fix.start = start
        env_fix.video_start_shoot = start - 2
        env_fix.SEQ_CHUNKS = val_all_chunks[seq_id]
        env_fix.SEQ_ID = SEQ_START + seq_id
        env.video_chunk_counter = 0
        env_fix.F1 = []
        env_fix.lag = []
        env_fix.Reward = []
        f1, lag, reward, state = evaluation_Multi(model, env_fix, loader, video_encoding_time, state)
        start = env_fix.start
        f1_sum += f1
        lag_sum += lag
        reward_sum += reward

        # f1_avg = f1
        # lag_avg = lag
        # reward_avg = reward

    f1_avg = f1_sum / SEQ_TOTAL
    lag_avg = lag_sum / SEQ_TOTAL
    reward_avg = reward_sum / SEQ_TOTAL
    print(epoch, cooked_name, f1_avg, lag_avg, reward_avg)
    log_file.write(str(int(epoch)) + '\t' +
                   str(f1_avg) + '\t' +
                   str(lag_avg) + '\t' +
                   str(reward_avg) + '\n')
    log_file.flush()

    add_str = 'Student1'
    model_save_path = f'Results/Fusion/Student1/' + "/%s_%d.model" % (add_str, int(epoch))
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(shared_model.state_dict(), model_save_path)
