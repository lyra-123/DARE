import numpy as np
import os
import torch
import env_fix
from ILCAS import DiscriminatorNet, Actor
from utils import load_one_trace

S_INFO = 8 # 8个状态量
S_LEN = 8  # past 8, 考虑过去8个块的历史数据
L = 2
H4, W4 = 36, 36
FRAMES = {
    0: [50, 25, 16, 10],
    1: [40, 20, 13, 8]
}

SUMMARY_DIR = 'Results'
LOG_FILE_VALID = 'Results/test_results/log_valid'
TEST_LOG_FOLDER_VALID = 'Results/test_results1/'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(model_actor, net_env, state, chunk_motion_map, video_encoding_time, current_video_id, seq_id):
    # state = np.zeros((S_INFO-1, S_LEN))
    # state = torch.from_numpy(state)
    # motion_map = np.zeros((H4, W4), np.float32)
    # motion_map = torch.from_numpy(motion_map)
    # motion_map = torch.zeros((H4, W4), dtype=torch.float32, device=device)
    # reward_sum = 0

    end_of_video = False
    while not end_of_video:
        # actor_motion_feat = actor_chunk_feat[net_env.video_chunk_counter]
        motion_map = torch.from_numpy(chunk_motion_map[net_env.video_chunk_counter]).float()
        # motion_map.copy_(torch.from_numpy(chunk_motion_map[net_env.video_chunk_counter]).float().to(device))
        with torch.no_grad():
            prob = model_actor(state.unsqueeze(0).float().to(device), motion_map.unsqueeze(0).float().to(device))
            # prob = model_actor(state.unsqueeze(0).type(dtype), motion_map.unsqueeze(0).type(dtype))
        action = prob.multinomial(num_samples=1).detach()
        # action = prob.argmax(dim=1)
        # knob = int(action.squeeze().cpu().numpy())
        knob = int(action.item())
        # with torch.no_grad():
        #     d_out = model_disc(state.unsqueeze(0).type(dtype), motion_map.unsqueeze(0).type(dtype),
        #                        action.squeeze(1).type(dlongtype))

        qp = knob // 20  # 0 to 10, because 80 // 16 = 5
        remainder = knob % 20
        skip = remainder // 5  # 0 to 3
        re = remainder % 5  # 0 to 3
        et = video_encoding_time[knob]

        bw, latency, buffer_size, size, f1, end_of_video = net_env.get_video_chunk(qp, skip, re, et)
        # with open(f'Results/test_results/log_test_5.txt', 'a') as f:
        #     f.write(f'{current_video_id}    {seq_id}    {net_env.video_chunk_counter-1}    {f1}    {latency}   {net_env.Reward[-1]}    {net_env.bw_use[-1]}\n')
        # motion_map = torch.from_numpy(feature_map)
        # motion_map.copy_(torch.tensor(feature_map, dtype=torch.float32, device=device))

        # dequeue history record
        # state = torch.roll(state, shifts=-1, dims=1)
        state = np.roll(state, -1, axis=1)
        state[0, -1] = size
        state[1, -1] = bw
        state[2, -1] = latency
        state[3, -1] = qp
        state[4, -1] = skip
        state[5, -1] = re
        state[6, -1] = buffer_size
        state = torch.from_numpy(state)

    f1_mean = np.mean(net_env.F1)
    # f1_std = np.std(net_env.F1)
    lag_mean = np.mean(net_env.lag)
    # lag_std = np.std(net_env.lag)
    Reward_mean = np.mean(net_env.Reward)
    Bit_mean = np.mean(net_env.bw_use)
    return f1_mean, lag_mean, Reward_mean, Bit_mean, state


def valid(shared_model_actor, epoch, log_file, val_all_chunks, merged_data_map, all_datasets_times, video_motion_map, cooked_bw, cooked_name, val_wl):
    # 检查文件夹是否存在，如果存在则删除
    if os.path.exists(TEST_LOG_FOLDER_VALID):
        os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    # os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model_actor = Actor().type(dtype)
    model_actor.eval()
    model_actor.load_state_dict(shared_model_actor.state_dict())

    # model_actor = Actor().to(device)
    # model_actor.eval()
    # actor_state_dict = getattr(shared_model_actor, "_orig_mod", shared_model_actor).state_dict()
    # model_actor.load_state_dict(actor_state_dict)

    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/', 4)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/Can/utils/network_range/32-48/', 2)
    # cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', 0)
    SEQ_TOTAL = 60  # 86400 25
    dataset_map = [4, 7, 10]
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    bit_sum = 0.0
    start = 4
    env = env_fix.Environment(cooked_bw=cooked_bw, seq_chunk_data=merged_data_map, start=start)
    state = np.zeros((S_INFO - 1, S_LEN))
    state = torch.from_numpy(state)
    current_video_id = 0
    for seq_id in range(SEQ_TOTAL):
        if current_video_id < 3 and seq_id == dataset_map[current_video_id]:
            # SEQ_START += dataset_pass_num[current_video_id]
            current_video_id += 1
        # if val_wl is not None and seq_id not in val_wl:
        #     continue
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
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=start, seq_chunks=seq_chunks[SEQ_START+seq_id], seq_id=SEQ_START+seq_id)
        # chunk_motion_map = video_motion_map[SEQ_START+seq_id]
        # actor_chunk_feat = motion_feat_cache_actor[SEQ_START+seq_id]
        chunk_motion_map = video_motion_map[env.seq_id]
        f1, lag, reward, bit, state = evaluation(model_actor, env, state, chunk_motion_map, video_encoding_time, current_video_id, seq_id)
        f1_sum += f1
        lag_sum += lag
        reward_sum += reward
        bit_sum += bit
        if current_video_id == 3:
            with open(f'Results/test_results1/log_test_.txt', 'a') as f:
                f.write(f'{current_video_id}    {seq_id}    {f1}    {lag}   {reward}    {bit}\n')
        # f1_avg = f1
        # lag_avg = lag
        # reward_avg = reward
    f1_avg = f1_sum / SEQ_TOTAL
    lag_avg = lag_sum / SEQ_TOTAL
    reward_avg = reward_sum / SEQ_TOTAL
    bit_avg = bit_sum / SEQ_TOTAL
    with open('Results/test_results1/log_test.txt', 'a') as f:
        f.write(f'avg   {f1_avg}   {lag_avg}   {reward_avg}    {bit_avg}\n')
    print(epoch, cooked_name, f1_avg, lag_avg, reward_avg)
    # log_file.write(str(int(epoch)) + '\t' +
    #                str(f1_avg) + '\t' +
    #                str(lag_avg) + '\t' +
    #                str(reward_avg) + '\n')
    # log_file.flush()
    # add_str = 'ILCAS'
    # model_save_path = SUMMARY_DIR + "/%s_%d_%f_%f.model" % (add_str, int(epoch), f1_avg, lag_avg)
    # torch.save(shared_model_actor.state_dict(), model_save_path)

# 45 48 55 64 68 73 83 86 95 107