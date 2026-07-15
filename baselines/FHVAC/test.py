import time
from collections import deque
import numpy as np
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
from PPO import Actor
from IL import ILAgent
from Fusion import FusionActor
import env_fix
from utils import load_one_trace, count_accuracy
import scipy.stats as stats
import time

# QP = [23, 28, 33, 38, 43]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
FPS = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]

S_INFO = 8
S_LEN = 8  # past 8
L = 2
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}

# RL = 'Results/RL_1600.model'
# IL = 'Results/IL/IL_4000.model'

NAME = 'FHVAC'
SUMMARY_DIR = 'Results'
LOG_FILE_VALID = 'Results/test_results/log_valid'
TEST_LOG_FOLDER_VALID = 'Results/test_results/'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def evaluation(model, net_env, video_encoding_time, state):
    # state = np.zeros((S_INFO, S_LEN))
    # state = torch.from_numpy(state)
    # reward_sum = 0
    # done = True
    # last_knob = 40

    # model.load_state_dict(model.state_dict())
    # log_path = log_path_ini + '_' + file_name
    # log_file = open(log_path, 'w')
    # time_stamp = 0
    while True:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        _, bw, latency, buffer_size, size, dynamics, _, f1, end_of_video = net_env.get_video_chunk(qp, skip, re, et)

        # dequeue history record
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

        if end_of_video:
            f1_mean = np.mean(net_env.F1)
            f1_std = np.std(net_env.F1)
            lag_mean = np.mean(net_env.lag)
            lag_std = np.std(net_env.lag)
            rw_mean = np.mean(net_env.Reward)
            rw_std = np.std(net_env.Reward)
            return f1_mean, f1_std, lag_mean, lag_std, rw_mean, rw_std, state


def valid(shared_model, epoch, log_file, val_all_chunks, merged_data_map, all_datasets_times):
    # 检查文件夹是否存在，如果存在则删除
    if os.path.exists(TEST_LOG_FOLDER_VALID):
        os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    # os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    num = 4
    cooked_bw, cooked_name = load_one_trace('/home/ubuntu/lyra/CASVA/test_trace/4G/', num)
    SEQ_TOTAL = 25  # 86400
    dataset_map = [4, 7, 10]
    dataset_pass_num = [11, 8, 73]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)
    start = 2
    current_video_id = 0
    env = env_fix.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map, seq_chunks=0, start=start, seq_id=0)
    f1_avg, lag_avg, reward_avg = 0., 0., 0.
    for seq_id in range(SEQ_TOTAL):
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=36, seq_chunks=seq_chunks[SEQ_START+seq_id], seq_id=SEQ_START+seq_id)
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            # SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
        if current_video_id == 0 or current_video_id == 3:
            env.FRAME = FRAMES[0]
        else:
            env.FRAME = FRAMES[1]
        env.F1 = []
        env.lag = []
        env.Reward = []
        video_encoding_time = all_datasets_times[current_video_id]
        env.SEQ_CHUNKS = val_all_chunks[seq_id]
        env.seq_id = seq_id
        env.video_chunk_counter = 0
        f1, f1_std, lag, lag_std, rw, rw_std, state = evaluation(model, env, video_encoding_time, state)
        f1_sum += f1
        lag_sum += lag
        reward_sum += rw
        with open(f'Results/test_results/medium/{num}.txt', 'a') as f:
            f.write(f'{current_video_id}    {seq_id}    {-1}    {f1}    {lag}   {rw}\n')

    f1_avg = f1_sum / SEQ_TOTAL
    lag_avg = lag_sum / SEQ_TOTAL
    reward_avg = reward_sum / SEQ_TOTAL
    print(epoch, cooked_name, f1_avg, lag_avg, reward_avg)
    log_file.write(str(int(epoch)) + '\t' +
                   str(f1_avg) + '\t' +
                   str(lag_avg) + '\t' +
                   str(reward_avg) + '\n')
    log_file.flush()
    add_str = 'RL'
    output_path  = SUMMARY_DIR +'/%s' % add_str
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_save_path = output_path + "/%s_%d.model" % (add_str, int(epoch))
    torch.save(shared_model.state_dict(), model_save_path)

def evaluation_IL(model, net_env, video_encoding_time, state):
    # state = np.zeros((S_INFO, S_LEN))
    # state = torch.from_numpy(state)
    # reward_sum = 0
    # done = True
    # last_knob = 40

    # model.load_state_dict(model.state_dict())
    # log_path = log_path_ini + '_' + file_name
    # log_file = open(log_path, 'w')
    # time_stamp = 0
    while True:
        with torch.no_grad():
            prob, _ = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        _, bw, latency, buffer_size, size, dynamics, _, Q, end_of_video = net_env.get_video_chunk(qp, skip, re, et)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        state[0, -1] = bw
        state[1, -1] = latency
        state[2, -1] = buffer_size
        state[3, -1] = qp
        state[4, -1] = skip
        state[5, -1] = re
        state[6, -1] = Q
        state = torch.from_numpy(state)

        if end_of_video:
            f1_mean = np.mean(net_env.F1)
            lag_mean = np.mean(net_env.lag)
            rw_mean = np.mean(net_env.Reward)
            return f1_mean, lag_mean, rw_mean, state

def valid_IL(shared_model, val_all_chunks, merged_data_map, all_datasets_times):
    # 检查文件夹是否存在，如果存在则删除
    if os.path.exists(TEST_LOG_FOLDER_VALID):
        os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    # os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = ILAgent().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',0)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    cooked_bw, cooked_name = load_one_trace('/home/ubuntu/lyra/CASVA/test_trace/4G/', 4)
    SEQ_TOTAL = 25  # 86400
    dataset_map = [4, 7, 10]
    dataset_pass_num = [11, 8, 73]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    state = np.zeros((S_INFO-1, S_LEN))
    state = torch.from_numpy(state)
    start = 2
    current_video_id = 0
    env = env_fix.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map, seq_chunks=0, start=start, seq_id=0)
    f1_avg, lag_avg, reward_avg = 0., 0., 0.
    for seq_id in range(SEQ_TOTAL):
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=36, seq_chunks=seq_chunks[SEQ_START+seq_id], seq_id=SEQ_START+seq_id)
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            # SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
        if current_video_id == 0 or current_video_id == 3:
            env.FRAME = FRAMES[0]
        else:
            env.FRAME = FRAMES[1]
        env.F1 = []
        env.lag = []
        env.Reward = []
        video_encoding_time = all_datasets_times[current_video_id]
        env.SEQ_CHUNKS = val_all_chunks[seq_id]
        env.seq_id = seq_id
        env.video_chunk_counter = 0
        f1, lag, rw, state = evaluation_IL(model, env, video_encoding_time, state)
        f1_sum += f1
        lag_sum += lag
        reward_sum += rw

    f1_avg = f1_sum / SEQ_TOTAL
    lag_avg = lag_sum / SEQ_TOTAL
    reward_avg = reward_sum / SEQ_TOTAL
    print(cooked_name, f1_avg, lag_avg, reward_avg)


def RL_test(test_model, index, start, chunk, total):
    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(torch.load(test_model))

    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', index)

    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk, total=total, chunk_start=0)

    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)

    while True:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))

        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        # print(qp, skip, re)
        _, bw, latency, buffer_size, size, dynamics, _, _, end_of_video = env.get_video_chunk(qp, skip, re)

        # dequeue history record
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

        if end_of_video:
            f1_mean = np.mean(env.F1)
            f1_std = np.std(env.F1, ddof=1)
            f1_standard_error = f1_std / np.sqrt(len(env.F1))
            f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)

            lag_mean = np.mean(env.lag)
            lag_std = np.std(env.lag, ddof=1)
            lag_standard_error = lag_std / np.sqrt(len(env.lag))
            lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)

            print(f1_mean, lag_mean)
            return f1_mean, lag_mean
            # with open('test.txt', 'a', newline='') as file:
            #     file.write(f'{f1_mean} {lag_mean} {f1_interval} {lag_interval}\n')


def IL_test(test_model, index, start, chunk, total):
    model = ILAgent().type(dtype)
    model.eval()
    model.load_state_dict(torch.load(test_model))

    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', index)

    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk, total=total, chunk_start=0)

    state = np.zeros((7, S_LEN))
    state = torch.from_numpy(state)

    while True:
        with torch.no_grad():
            prob, _ = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 16  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 16
        skip = remainder // 4  # 0 to 3
        re = remainder % 4  # 0 to 3

        _, bw, latency, buffer_size, size, dynamics, _, Q, end_of_video = env.get_video_chunk(qp, skip, re)

        # dequeue history record
        state = np.roll(state, -1, axis=1)
        state[0, -1] = bw
        state[1, -1] = latency
        state[2, -1] = buffer_size
        state[3, -1] = QP[qp]
        state[4, -1] = FPS[skip]
        state[5, -1] = RE[re]
        state[6, -1] = Q
        state = torch.from_numpy(state)

        if end_of_video:
            f1_mean = np.mean(env.F1)
            f1_std = np.std(env.F1, ddof=1)
            f1_standard_error = f1_std / np.sqrt(len(env.F1))
            f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)

            lag_mean = np.mean(env.lag)
            lag_std = np.std(env.lag, ddof=1)
            lag_standard_error = lag_std / np.sqrt(len(env.lag))
            lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)

            print(f1_mean, lag_mean)
            return f1_mean, lag_mean
            # with open('test.txt', 'a', newline='') as file:
            #     file.write(f'{f1_mean} {lag_mean} {f1_interval} {lag_interval}\n')


def evaluation_F(RL_model, IL_model, fusion_actor, net_env, video_encoding_time, state_RL, state_IL, current_video_id, seq_id):
    # state_RL = np.zeros((8, S_LEN))
    # state_RL = torch.from_numpy(state_RL)
    # state_IL = np.zeros((7, S_LEN))
    # state_IL = torch.from_numpy(state_IL)
    # done = True
    # last_knob = 40
    end_of_video = False
    while not end_of_video:
        with torch.no_grad():
            frl = RL_model.get_feature(state_RL.unsqueeze(0).type(dtype))  # 获取RL特征
            fil = IL_model.get_feature(state_IL.unsqueeze(0).type(dtype))  # 获取IL特征
            prob = fusion_actor(frl, fil)
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        _, bw, latency, buffer_size, size, dynamics, f1, Q, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
        # with open(f'Results/test_results/log_test_00.txt', 'a') as f:
        #     f.write(f'{current_video_id}    {seq_id}    {net_env.video_chunk_counter-1}    {f1}    {latency}   {net_env.Reward[-1]}    {net_env.bw_use[-1]}\n')


        state_RL = np.roll(state_RL, -1, axis=1)
        state_RL[0, -1] = bw
        state_RL[1, -1] = latency
        state_RL[2, -1] = buffer_size
        state_RL[3, -1] = qp
        state_RL[4, -1] = skip
        state_RL[5, -1] = re
        state_RL[6, -1] = size
        state_RL[7, -1] = dynamics
        state_RL = torch.from_numpy(state_RL)

        state_IL = np.roll(state_IL, -1, axis=1)
        state_IL[0, -1] = bw
        state_IL[1, -1] = latency
        state_IL[2, -1] = buffer_size
        state_IL[3, -1] = qp
        state_IL[4, -1] = skip
        state_IL[5, -1] = re
        state_IL[6, -1] = Q
        state_IL = torch.from_numpy(state_IL)

    f1_mean = np.mean(net_env.F1)
    lag_mean = np.mean(net_env.lag)
    Reward_mean = np.mean(net_env.Reward)
    Bit_mean = np.mean(net_env.bw_use)
    return f1_mean, lag_mean, Reward_mean, Bit_mean, state_RL, state_IL


def valid_fusion(RLmodel, ILmodel, fusion_actor, epoch, log_file, val_all_chunks, merged_data_map, all_datasets_times):
    RL_model = Actor().type(dtype)
    RL_model.eval()
    RL_model.load_state_dict(RLmodel.state_dict())

    IL_model = ILAgent().type(dtype)
    IL_model.eval()
    IL_model.load_state_dict(ILmodel.state_dict())

    model = FusionActor().type(dtype)
    model.eval()
    model.load_state_dict(fusion_actor.state_dict())

    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/traces/', 118)
    num = 4
    cooked_bw, cooked_name = load_one_trace('/home/ubuntu/lyra/CASVA/test_trace/4G/', num)
    SEQ_TOTAL = 60  # 86400
    SEQ_START = 0
    dataset_map = [4, 7, 10]
    dataset_pass_num = [11, 8, 73]
    start = 4
    current_video_id = 0
    env = env_fix.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map, seq_chunks=0, start=start, seq_id=0)
    state_RL = np.zeros((8, S_LEN))
    state_RL = torch.from_numpy(state_RL)
    state_IL = np.zeros((7, S_LEN))
    state_IL = torch.from_numpy(state_IL)
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    bit_sum = 0.0
    f1_avg, lag_avg, reward_avg = 0., 0., 0.
    for seq_id in range(SEQ_TOTAL):
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=0, seq_chunks=seq_chunks[SEQ_START+seq_id], seq_id=SEQ_START+seq_id)
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            # SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
        if current_video_id == 0 or current_video_id == 3:
            env.FRAME = FRAMES[0]
        else:
            env.FRAME = FRAMES[1]

        env.F1 = []
        env.lag = []
        env.Reward = []
        env.bw_use = []
        video_encoding_time = all_datasets_times[current_video_id]
        env.SEQ_CHUNKS = val_all_chunks[seq_id]
        env.seq_id = seq_id
        env.video_chunk_counter = 0
        f1, lag, reward, bit, state_RL, state_IL = evaluation_F(RL_model, IL_model, model, env, video_encoding_time, state_RL, state_IL, current_video_id, seq_id)
        f1_sum += f1
        lag_sum += lag
        reward_sum += reward
        bit_sum += bit
        if current_video_id == 3:
            with open(f'Results/test_results/log_test_.txt', 'a') as f:
                f.write(f'{current_video_id}    {seq_id}    {f1}    {lag}   {reward}    {bit}\n')

    f1_avg = f1_sum / SEQ_TOTAL
    lag_avg = lag_sum / SEQ_TOTAL
    reward_avg = reward_sum / SEQ_TOTAL
    bit_avg = bit_sum / SEQ_TOTAL
    # with open('Results/test_results/log_test.txt', 'a') as f:
    #     f.write(f'avg   {f1_avg}   {lag_avg}   {reward_avg}    {bit_avg}\n')
    print(epoch, cooked_name, f1_avg, lag_avg, reward_avg)
    # log_file.write(str(int(epoch)) + '\t' +
    #                str(f1_avg) + '\t' +
    #                str(lag_avg) + '\t' +
    #                str(reward_avg) + '\n')
    # log_file.flush()
    #
    # add_str = 'IL'
    # model_save_path = f'Results/Fusion/{NAME}' + "/%s_%s_%d.model" % (add_str, NAME, int(epoch))
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # torch.save(ILmodel.state_dict(), model_save_path)
    #
    # add_str = 'RL'
    # model_save_path = f'Results/Fusion/{NAME}' + "/%s_%s_%d.model" % (add_str, NAME, int(epoch))
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # torch.save(RLmodel.state_dict(), model_save_path)
    #
    # add_str = 'Fusion'
    # model_save_path = f'Results/Fusion/{NAME}' + "/%s_%s_%d.model" % (add_str, NAME, int(epoch))
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # torch.save(fusion_actor.state_dict(), model_save_path)


def Fusion_test(RLmodel, ILmodel, fusion_actor, index, start, chunk, total, chunk_start):
    RL_model = Actor().type(dtype)
    RL_model.eval()
    RL_model.load_state_dict(torch.load(RLmodel))
    # RL_model.load_state_dict(RLmodel.state_dict())

    IL_model = ILAgent().type(dtype)
    IL_model.eval()
    IL_model.load_state_dict(torch.load(ILmodel))
    # IL_model.load_state_dict(ILmodel.state_dict())

    model = FusionActor().type(dtype)
    model.eval()
    model.load_state_dict(torch.load(fusion_actor))
    # model_block.load_state_dict(fusion_block.state_dict())
    # model.load_state_dict(fusion_actor.state_dict())

    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/', index)
    # print(NAME[name], cooked_name)

    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk, total=total, chunk_start=chunk_start)

    state_RL = np.zeros((8, S_LEN))
    state_RL = torch.from_numpy(state_RL)
    state_IL = np.zeros((7, S_LEN))
    state_IL = torch.from_numpy(state_IL)

    while True:
        frl = RL_model.get_feature(state_RL.unsqueeze(0).type(dtype))  # 获取RL特征
        fil = IL_model.get_feature(state_IL.unsqueeze(0).type(dtype))  # 获取IL特征
        prob = model(frl, fil)
        # prob_RL = RL_model(state_RL.unsqueeze(0).type(dtype))
        # combined_prob = torch.softmax(prob + prob_RL[0], dim=1)
        # action = combined_prob.multinomial(num_samples=1).detach()
        action = prob.multinomial(num_samples=1).detach()
        # prob_RL = RL_model(state_RL.unsqueeze(0).type(dtype))
        # prob_IL, _ = IL_model(state_IL.unsqueeze(0).type(dtype))
        # prob = model(prob_RL, prob_IL)
        # action = prob.multinomial(num_samples=1).detach()

        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 16  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 16
        skip = remainder // 4  # 0 to 3
        re = remainder % 4  # 0 to 3

        _, bw, latency, buffer_size, size, dynamics, _, Q, end_of_video = env.get_video_chunk(qp, skip, re)

        state_RL = np.roll(state_RL, -1, axis=1)
        state_RL[0, -1] = bw
        state_RL[1, -1] = latency
        state_RL[2, -1] = buffer_size
        state_RL[3, -1] = qp
        state_RL[4, -1] = skip
        state_RL[5, -1] = re
        state_RL[6, -1] = size
        state_RL[7, -1] = dynamics
        state_RL = torch.from_numpy(state_RL)

        state_IL = np.roll(state_IL, -1, axis=1)
        state_IL[0, -1] = bw
        state_IL[1, -1] = latency
        state_IL[2, -1] = buffer_size
        state_IL[3, -1] = qp
        state_IL[4, -1] = skip
        state_IL[5, -1] = re
        state_IL[6, -1] = Q
        state_IL = torch.from_numpy(state_IL)

        if end_of_video:
            # print(prob)
            # exit()
            # f1_mean = np.mean(env.F1)
            # f1_std = np.std(env.F1, ddof=1)
            # f1_standard_error = f1_std / np.sqrt(len(env.F1))
            # f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)
            #
            # lag_mean = np.mean(env.lag)
            # lag_std = np.std(env.lag, ddof=1)
            # lag_standard_error = lag_std / np.sqrt(len(env.lag))
            # lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)
            #
            # print(f1_mean, lag_mean)
            # return f1_mean, lag_mean
            return (env.F1, env.lag, env.bw_use, env.Reward, cooked_name,
                    env.lag_1, env.lag_2, env.lag_3, env.lag_4, env.lag_5)
            # with open('test.txt', 'a', newline='') as file:
            #     file.write(f'{f1_mean} {lag_mean} {f1_interval} {lag_interval}\n')
