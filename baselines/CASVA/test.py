import time
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PPO import Actor
import env_fix
from utils import load_one_trace, C_R
import scipy.stats as stats
import time

S_INFO = 8 # 8个状态量
S_LEN = 8  # past 8, 考虑过去8个块的历史数据
L = 2
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}
NAME = ('DETRAC', 'LMOT', 'D²-City_1080p', 'D²-City_720p', 'DSEC')

SUMMARY_DIR = 'Results'
# LOG_FILE = 'Results/log'
# TEST_LOG_FOLDER = 'Results/test_results/'
LOG_FILE_VALID = 'Results/test_results/log_valid'
TEST_LOG_FOLDER_VALID = 'Results/test_results/'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def evaluation(model, net_env, video_encoding_time, state, current_video_id, seq_id):
    # state = np.zeros((S_INFO, S_LEN))
    # state = torch.from_numpy(state)
    # reward_sum = 0
    # done = True
    # last_knob = 40

    end_of_video = False
    while not end_of_video:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5 # 0 to 3
        et = video_encoding_time[knob]

        bw, latency, buffer_size, size, dynamics, f1, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
        # with open(f'Results/test_results/log_test_00.txt', 'a') as f:
        #     f.write(f'{current_video_id}    {seq_id}    {net_env.video_chunk_counter-1}    {f1}    {latency}   {net_env.Reward[-1]}    {net_env.bw_use[-1]}\n')

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

    f1_mean = np.mean(net_env.F1)
    lag_mean = np.mean(net_env.lag)
    Reward_mean = np.mean(net_env.Reward)
    Bit_mean = np.mean(net_env.bw_use)
    return f1_mean, lag_mean, Reward_mean, Bit_mean, state


def valid(shared_model, epoch, log_file, val_all_chunks, merged_data_map_val, all_datasets_times):
    # # 检查文件夹是否存在，如果存在则删除
    # if os.path.exists(TEST_LOG_FOLDER_VALID):
    #     os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    # # os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    # os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    num = 4
    cooked_bw, cooked_name = load_one_trace('/home/ubuntu/lyra/CASVA/test_trace/4G/', num)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    SEQ_TOTAL = 60 # 86400 25
    SEQ_START = 0
    dataset_map = [4, 7, 10]
    dataset_pass_num = [11, 8, 73]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    bit_sum = 0.0
    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)
    start = 4
    current_video_id = 0
    f1_avg, lag_avg, reward_avg = 0., 0., 0.
    env = env_fix.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map_val, seq_chunks=0, start=start, seq_id=0)
    for seq_id in range(SEQ_TOTAL):
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
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=36, seq_chunks=seq_chunks[SEQ_START+seq_id], seq_id=SEQ_START+seq_id)
        f1, lag, reward, bit, state = evaluation(model, env, video_encoding_time, state, current_video_id, seq_id)
        f1_sum += f1
        lag_sum += lag
        reward_sum += reward
        bit_sum += bit
        if current_video_id == 3:
            with open(f'Results/test_results/log_test_.txt', 'a') as f:
                f.write(f'{current_video_id}    {seq_id}    {f1}    {lag}   {reward}    {bit}\n')

        # f1_avg = f1
        # lag_avg = lag
        # reward_avg = reward

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
    # add_str = 'CASVA'
    # model_save_path = SUMMARY_DIR + "/%s_%d_%f_%f.model" % (add_str, int(epoch), f1_avg, lag_avg)
    # torch.save(shared_model.state_dict(), model_save_path)


def test(test_model, index, start, chunk, total, chunk_start):
    model = Actor().type(dtype)
    model.eval()
    model.load_state_dict(torch.load(test_model,weights_only=True))

    cooked_bw, cooked_name = load_one_trace('test_trace/4G/', index)
    # print(NAME[name], cooked_name)

    env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk,
                              total=total, chunk_start=chunk_start)

    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)

    while True:
        with torch.no_grad():
            prob = model(state.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        knob = int(action.squeeze().cpu().numpy())

        qp = knob // 16  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 16
        skip = remainder // 4  # 0 to 3
        re = remainder % 4  # 0 to 3

        bw, latency, buffer_size, size, dynamics, f1, end_of_video = env.get_video_chunk(qp, skip, re)

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

            # f1_mean = np.mean(env.F1)
            # f1_std = np.std(env.F1, ddof=1)
            # f1_standard_error = f1_std / np.sqrt(len(env.F1))
            # f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)
            #
            # lag_mean = np.mean(env.lag)
            # lag_std = np.std(env.lag, ddof=1)
            # lag_standard_error = lag_std / np.sqrt(len(env.lag))
            # lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)

            return (env.F1, env.lag, env.bw_use, env.Reward, cooked_name,
                    env.lag_1, env.lag_2, env.lag_3, env.lag_4, env.lag_5)
            # with open('test.txt', 'a', newline='') as file:
            #     file.write(f'{f1_mean} {lag_mean} {f1_interval} {lag_interval}\n')
            # break