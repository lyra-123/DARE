# -*- coding: utf-8 -*-

import os
import time
import pickle
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict

SENSITIVITY_CSV={
    "DETRAC":"../sensitivity/DETRAC_sensitivity.csv",
    "DSEC":"../sensitivity/DSEC_sensitivity.csv",
    "LMOT":"../sensitivity/LMOT_sensitivity.csv",
    "D²-City":"../sensitivity/D²-City_sensitivity.csv"
}

FEATURE_H5={
    "DETRAC":"../deg_feats/layer1/layer1/DETRAC_layer1.h5",
    "DSEC":"../deg_feats/layer1/layer1/DSEC_layer1.h5",
    "LMOT":"../deg_feats/layer1/layer1/LMOT_layer1.h5",
    "D²-City":"../deg_feats/layer1/layer1/D²-City_layer1.h5"
}

DATASET_NAMES=["DETRAC","DSEC","LMOT","D²-City"]
SAVE_DIR="results/labels_to_cluster/2"
DELTA=0.003

def load_sensitivity(csv_dict,dataset_names):
    sensitivity_dict={}
    for ds_name in dataset_names:
        df=pd.read_csv(csv_dict[ds_name])
        required={"seq","s_qp","s_skip","s_re"}
        missing=required-set(df.columns)
        if missing:
            raise ValueError(f"{ds_name}: sensitivity CSV 缺少列 {missing}")
        ds_sensitivity={}
        for _,row in df.iterrows():
            seq=int(row["seq"])
            ds_sensitivity[seq]=np.array([row["s_qp"],row["s_skip"],row["s_re"]],dtype=np.float32)
        sensitivity_dict[ds_name]=ds_sensitivity
        print(f"{ds_name}: {len(ds_sensitivity)} sessions")
    return sensitivity_dict

def load_features(feature_h5_dict,dataset_names):
    feat_dict={}
    for ds_name in dataset_names:
        ds_feats={}
        with h5py.File(feature_h5_dict[ds_name],"r") as f:
            for vid in sorted(f.keys(),key=lambda x:int(x)):
                feats=f[vid]["features"][:]
                for chunk_idx,feat in enumerate(feats):
                    if feat.ndim==4:
                        p1_vec=feat[0].mean(axis=(-1,-2))
                        diff_vec=feat[2].mean(axis=(-1,-2))
                    elif feat.ndim==1 and feat.shape[0]==128:
                        p1_vec=feat[:64]
                        diff_vec=feat[64:]
                    else:
                        raise ValueError(f"{ds_name} video={vid} chunk={chunk_idx}: 无法识别特征形状 {feat.shape}")
                    p1_vec=np.asarray(p1_vec,dtype=np.float32).reshape(-1)
                    diff_vec=np.asarray(diff_vec,dtype=np.float32).reshape(-1)
                    if len(p1_vec)!=64 or len(diff_vec)!=64:
                        raise ValueError(f"{ds_name} video={vid} chunk={chunk_idx}: p1和diff应分别为64维")
                    ds_feats[(int(vid),int(chunk_idx))]=(p1_vec,diff_vec)
        feat_dict[ds_name]=ds_feats
        print(f"{ds_name}: {len(ds_feats)} chunks")
    return feat_dict

def build_session_samples(sensitivity_dict,feat_dict,dataset_names):
    X=[]
    R=[]
    meta=[]
    for ds_name in dataset_names:
        groups=defaultdict(list)
        for (video_id,chunk_idx),(p1_vec,diff_vec) in feat_dict[ds_name].items():
            groups[int(video_id)].append((int(chunk_idx),p1_vec,diff_vec))
        for video_id,chunks in sorted(groups.items()):
            if video_id not in sensitivity_dict[ds_name]:
                continue
            chunks=sorted(chunks,key=lambda x:x[0])
            p1=np.mean(np.array([x[1] for x in chunks],dtype=np.float32),axis=0)
            diff=np.mean(np.array([x[2] for x in chunks],dtype=np.float32),axis=0)
            X.append(np.concatenate([p1,diff],axis=0))
            R.append(sensitivity_dict[ds_name][video_id])
            meta.append((ds_name,video_id))
    X=np.asarray(X,dtype=np.float32)
    R=np.asarray(R,dtype=np.float32)
    if len(X)==0:
        raise ValueError("没有成功对齐 sensitivity 和 degradation feature")
    print(f"session samples: {len(X)}, feature dim: {X.shape[1]}")
    return X,R,meta

def calc_variance(R):
    if len(R)<=1:
        return 0.0
    return float(np.sum(np.var(R,axis=0,ddof=0)))

def find_best_split(X,R,indices):
    X_node=X[indices]
    n=len(indices)
    best_split=None
    best_j=np.inf

    for feature_id in range(X.shape[1]):
        values=X_node[:,feature_id]
        unique_values=np.unique(values)
        if len(unique_values)<2:
            continue

        thresholds=(unique_values[:-1]+unique_values[1:])/2.0
        for threshold in thresholds:
            left_mask=values<=threshold
            right_mask=values>threshold
            left_indices=indices[left_mask]
            right_indices=indices[right_mask]

            j_left=calc_variance(R[left_indices])
            j_right=calc_variance(R[right_indices])
            j=(len(left_indices)/n)*j_left+(len(right_indices)/n)*j_right

            if j<best_j:
                best_j=j
                best_split=(int(feature_id),float(threshold),left_indices,right_indices)

    return best_split,best_j

def build_tree(X,R,indices,delta):
    j_c=calc_variance(R[indices])
    best_split,j_m=find_best_split(X,R,indices)

    if best_split is None:
        return {"is_leaf":True}

    gain=j_c-j_m
    if gain<delta:
        return {"is_leaf":True}

    feature_id,threshold,left_indices,right_indices=best_split
    node={
        "is_leaf":False,
        "feature_id":feature_id,
        "threshold":threshold,
        "gain":float(gain)
    }
    node["left"]=build_tree(X,R,left_indices,delta)
    node["right"]=build_tree(X,R,right_indices,delta)
    return node

def assign_leaf_ids(node,next_id=0):
    if node["is_leaf"]:
        node["cluster_id"]=next_id
        return next_id+1
    next_id=assign_leaf_ids(node["left"],next_id)
    next_id=assign_leaf_ids(node["right"],next_id)
    return next_id

def predict_one(tree,x):
    node=tree
    while not node["is_leaf"]:
        if x[node["feature_id"]]<=node["threshold"]:
            node=node["left"]
        else:
            node=node["right"]
    return int(node["cluster_id"])

def predict(tree,X):
    return np.array([predict_one(tree,x) for x in X],dtype=np.int32)

def save_labels(meta,labels,save_dir):
    rows=[]
    for i,(ds_name,video_id) in enumerate(meta):
        rows.append({"dataset":ds_name,"seq":int(video_id),"cluster_id":int(labels[i])})
    out_csv=os.path.join(save_dir,"session_cluster_labels.csv")
    pd.DataFrame(rows).sort_values(["cluster_id","dataset","seq"]).to_csv(out_csv,index=False,encoding="utf-8-sig")
    return out_csv

def run(delta=DELTA,save_dir=SAVE_DIR):
    os.makedirs(save_dir,exist_ok=True)

    sensitivity_dict=load_sensitivity(SENSITIVITY_CSV,DATASET_NAMES)
    feat_dict=load_features(FEATURE_H5,DATASET_NAMES)
    X,R,meta=build_session_samples(sensitivity_dict,feat_dict,DATASET_NAMES)

    indices=np.arange(len(X),dtype=np.int32)
    tree=build_tree(X,R,indices,delta)
    n_clusters=assign_leaf_ids(tree)
    labels=predict(tree,X)

    model={"tree":tree,"delta":float(delta),"feature_dim":int(X.shape[1]),"n_clusters":int(n_clusters)}
    with open(os.path.join(save_dir,"tree_model.pkl"),"wb") as f:
        pickle.dump(model,f)

    out_csv=save_labels(meta,labels,save_dir)

    print(f"clusters: {n_clusters}")
    print(f"cluster counts: {dict(zip(*np.unique(labels,return_counts=True)))}")
    return model,out_csv

if __name__=="__main__":
    start=time.perf_counter()
    run(delta=DELTA,save_dir=SAVE_DIR)
    print(f"training time: {time.perf_counter()-start:.2f}s")
