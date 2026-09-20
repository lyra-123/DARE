# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from network import ImitationNet

S_INFO=4
S_LEN=8
C=64
SWITCH_K=3

dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

TREE_PATH="decision_tree/model_data/2/tree_cluster.pkl"
MODEL_DIR="encoding_policy/model_data"

def load_tree(tree_path):
    with open(tree_path,"rb") as f:
        tree_info=pickle.load(f)
    tree=tree_info["tree"]
    n_clusters=int(tree_info["n_clusters"])
    feature_dim=int(tree_info["feature_dim"])
    if feature_dim!=128:
        raise ValueError(f"决策树输入维度应为128，当前为{feature_dim}")
    return tree,n_clusters

def predict_cluster(tree,deg_feat):
    deg_feat=np.asarray(deg_feat,dtype=np.float32).reshape(-1)
    if len(deg_feat)!=128:
        raise ValueError(f"退化特征应为128维，当前为{len(deg_feat)}维")
    node=tree
    while not node["is_leaf"]:
        feature_id=node["feature_id"]
        threshold=node["threshold"]
        node=node["left"] if deg_feat[feature_id]<=threshold else node["right"]
    return int(node["cluster_id"])

def load_policy_models(model_dir,n_clusters):
    models={}
    for cluster_id in range(n_clusters):
        model_path=os.path.join(model_dir,f"{cluster_id}.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到 cluster {cluster_id} 的模型: {model_path}")
        model=ImitationNet(C=C,k=S_LEN).type(dtype)
        state_dict=torch.load(model_path,map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict)
        model.eval()
        models[cluster_id]=model
    return models

class OnlineSwitcher:
    def __init__(self,tree,models,switch_k=SWITCH_K):
        self.tree=tree
        self.models=models
        self.switch_k=switch_k
        self.active_cluster=None
        self.candidate_cluster=None
        self.candidate_count=0

    def reset(self):
        self.active_cluster=None
        self.candidate_cluster=None
        self.candidate_count=0

    def update(self,deg_feat):
        predicted_cluster=predict_cluster(self.tree,deg_feat)

        if predicted_cluster not in self.models:
            raise KeyError(f"cluster {predicted_cluster} 没有对应的决策模型")

        if self.active_cluster is None:
            self.active_cluster=predicted_cluster
            self.candidate_cluster=None
            self.candidate_count=0
            return self.active_cluster,predicted_cluster,False

        if predicted_cluster==self.active_cluster:
            self.candidate_cluster=None
            self.candidate_count=0
            return self.active_cluster,predicted_cluster,False

        if predicted_cluster==self.candidate_cluster:
            self.candidate_count+=1
        else:
            self.candidate_cluster=predicted_cluster
            self.candidate_count=1

        switched=False
        if self.candidate_count>=self.switch_k:
            self.active_cluster=self.candidate_cluster
            self.candidate_cluster=None
            self.candidate_count=0
            switched=True

        return self.active_cluster,predicted_cluster,switched

    def get_model(self):
        if self.active_cluster is None:
            raise RuntimeError("尚未确定当前 cluster")
        return self.models[self.active_cluster]

def online_evaluation(tree,models,net_env,merged_deg_map,video_encoding_time,switch_k=SWITCH_K):
    switcher=OnlineSwitcher(tree,models,switch_k)
    state=torch.zeros((S_INFO,S_LEN))
    p1_state=torch.zeros((C,S_LEN+1))
    diff_state=torch.zeros((C,S_LEN+1))

    while True:
        key=(net_env.seq_id,net_env.video_chunk_counter)
        if key not in merged_deg_map:
            raise KeyError(f"未找到退化特征: {key}")

        deg_feat=np.asarray(merged_deg_map[key],dtype=np.float32).reshape(-1)
        if len(deg_feat)!=128:
            raise ValueError(f"{key}: 退化特征应为128维，当前为{len(deg_feat)}维")

        active_cluster,predicted_cluster,switched=switcher.update(deg_feat)

        if switched:
            state.zero_()
            p1_state.zero_()
            diff_state.zero_()

        p1_feat=deg_feat[:C]
        diff_feat=deg_feat[C:]

        p1_state=torch.roll(p1_state,-1,dims=1)
        diff_state=torch.roll(diff_state,-1,dims=1)
        p1_state[:,-1]=torch.from_numpy(p1_feat)
        diff_state[:,-1]=torch.from_numpy(diff_feat)

        model=switcher.get_model()

        with torch.no_grad():
            qp_logits,s_logits,r_logits=model(
                state.unsqueeze(0).type(dtype),
                p1_state.unsqueeze(0).type(dtype),
                diff_state.unsqueeze(0).type(dtype)
            )
            qp=int(torch.argmax(qp_logits,dim=1).item())
            s=int(torch.argmax(s_logits,dim=1).item())
            r=int(torch.argmax(r_logits,dim=1).item())

        knob=qp*25+s*5+r
        encot_t=video_encoding_time[knob]

        bw,latency,buffer_size,size,_,_,end_of_video=net_env.get_video_chunk(qp,s,r,encot_t)

        state=torch.roll(state,-1,dims=1)
        state[0,-1]=bw
        state[1,-1]=qp
        state[2,-1]=s
        state[3,-1]=r

        if end_of_video:
            return np.mean(net_env.F1),np.mean(net_env.lag),np.mean(net_env.Reward)

def load_online_system(tree_path=TREE_PATH,model_dir=MODEL_DIR):
    tree,n_clusters=load_tree(tree_path)
    models=load_policy_models(model_dir,n_clusters)
    return tree,models

if __name__=="__main__":
    tree,models=load_online_system()
    print(f"online switching system loaded, clusters={len(models)}")
