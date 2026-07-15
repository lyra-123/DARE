import numpy as np
import os
import torch
from network import ImitationNet
import env_fix
from utils import load_one_trace

QP=[1,1.2174,1.4348,1.6522,1.8696]
FPS=[1.0,0.75,0.5,0.3333,0.1667]
RE=[1.0,0.4444,0.1667,0.0370]

S_INFO=4
D_INFO=128
S_LEN=8

FRAMES={0:[50,25,16,10],1:[40,20,13,8]}

NAME='IL'
SUMMARY_DIR='Results'
LOG_FILE_VALID='Results/test_results/log_valid'

dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def evaluation(model, net_env, merged_deg_map, video_encoding_time, state, deg_state, diff_state):
    while True:
        with torch.no_grad():
            deg_feat,diff_feat=merged_deg_map[(net_env.seq_id,net_env.video_chunk_counter)]
            deg_state=torch.roll(deg_state,-1,dims=1)
            diff_state=torch.roll(diff_state,-1,dims=1)
            deg_state[:,-1]=torch.tensor(deg_feat)
            diff_state[:,-1]=torch.tensor(diff_feat)
            prob=model(state.unsqueeze(0).type(dtype),deg_state.unsqueeze(0).type(dtype),diff_state.unsqueeze(0).type(dtype))
            action=torch.argmax(prob,dim=1)

        knob=int(action.item())
        qp=knob//25
        remainder=knob%25
        skip=remainder//5
        re=remainder%5
        et=video_encoding_time[knob]

        bw,latency,buffer_size,size,_,_,end_of_video=net_env.get_video_chunk(qp,skip,re,et)

        state=torch.roll(state,-1,dims=1)
        state[0,-1]=bw
        state[1,-1]=qp
        state[2,-1]=skip
        state[3,-1]=re

        if end_of_video:
            return np.mean(net_env.F1),np.mean(net_env.lag),np.mean(net_env.Reward),state,deg_state,diff_state



def valid(shared_model,epoch,log_file,val_all_chunks,merged_data_map,merged_deg_map,all_datasets_times,val_wl=None,val_config_list=None):

    model=ImitationNet().type(dtype)
    model.eval()
    model.load_state_dict(shared_model.state_dict())

    cooked_bw,cooked_name=load_one_trace('/home/dell/lyra/CASVA/test_trace/4G/',4)

    SEQ_TOTAL=len(val_all_chunks)

    dataset_map=[4,7,10]

    f1_sum=0
    lag_sum=0
    reward_sum=0

    state=torch.zeros((S_INFO,S_LEN))
    deg_state=torch.zeros((D_INFO,S_LEN+1))
    diff_state=torch.zeros((D_INFO,S_LEN+1))

    current_video_id=0
    valid_count=0

    env=env_fix.Environment(cooked_bw=cooked_bw,seq_chunk_data=merged_data_map,seq_chunks=0,start=2,seq_id=0)


    for seq_id in range(SEQ_TOTAL):
        if val_wl is not None and seq_id not in val_wl:
            continue
        valid_count+=1
        if current_video_id<3 and seq_id==dataset_map[current_video_id]:
            current_video_id+=1
        env.FRAME=FRAMES[0] if current_video_id==0 or current_video_id==3 else FRAMES[1]
        env.F1=[]
        env.lag=[]
        env.Reward=[]
        env.bw_use=[]
        video_encoding_time=all_datasets_times[current_video_id]
        env.SEQ_CHUNKS=val_all_chunks[seq_id]
        env.seq_id=seq_id
        env.video_chunk_counter=0
        f1,lag,rw,state,deg_state,diff_state=evaluation(model,env,merged_deg_map,video_encoding_time,state,deg_state,diff_state)
        f1_sum+=f1
        lag_sum+=lag
        reward_sum+=rw
    f1_avg=f1_sum/valid_count
    lag_avg=lag_sum/valid_count
    reward_avg=reward_sum/valid_count

    print(epoch,cooked_name,f1_avg,lag_avg,reward_avg)

    if log_file is not None:
        log_file.write(str(int(epoch))+'\t'+str(f1_avg)+'\t'+str(lag_avg)+'\t'+str(reward_avg)+'\n')
        log_file.flush()

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    torch.save(shared_model.state_dict(),SUMMARY_DIR+"/IL_%d.model"%int(epoch))