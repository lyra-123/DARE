import os
import h5py
import argparse
import numpy as np
import pandas as pd

def find_dataset_key(h5_path,candidates=None):
    if candidates is None:
        candidates=["f1","accuracy","acc","score"]
    keys=[]
    with h5py.File(h5_path,"r") as f:
        def collect(name,obj):
            if isinstance(obj,h5py.Dataset):
                keys.append(name)
        f.visititems(collect)
    for key in keys:
        low=key.lower()
        for cand in candidates:
            if cand.lower() in low:
                return key
    raise KeyError("F1 dataset key not found.")

def load_h5_dataset(h5_path,key):
    with h5py.File(h5_path,"r") as f:
        return np.asarray(f[key],dtype=np.float32)

def reshape_f1_to_grid(data,n_qp,n_skip,n_re):
    total=n_qp*n_skip*n_re
    if data.ndim>=3 and tuple(data.shape[-3:])==(n_qp,n_skip,n_re):
        return data.reshape(-1,n_qp,n_skip,n_re)
    if data.shape[-1]==total:
        return data.reshape(-1,total).reshape(-1,n_qp,n_skip,n_re)
    raise ValueError(f"Invalid shape: {data.shape}, expected {n_qp}×{n_skip}×{n_re}={total} configurations.")

def calc_sensitivity_one(x):
    v_qp=np.mean(np.var(x,axis=0,ddof=0))
    v_skip=np.mean(np.var(x,axis=1,ddof=0))
    v_re=np.mean(np.var(x,axis=2,ddof=0))
    s_qp=2.0*np.sqrt(v_qp)
    s_skip=2.0*np.sqrt(v_skip)
    s_re=2.0*np.sqrt(v_re)
    return s_qp,s_skip,s_re

def calc_sensitivity(f1_grid):
    rows=[]
    for i in range(f1_grid.shape[0]):
        s_qp,s_skip,s_re=calc_sensitivity_one(f1_grid[i])
        rows.append({
            "sample_id":i,
            "s_qp":float(s_qp),
            "s_skip":float(s_skip),
            "s_re":float(s_re)
        })
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--h5_path",type=str,required=True)
    parser.add_argument("--f1_key",type=str,default=None)
    parser.add_argument("--n_qp",type=int,default=5)
    parser.add_argument("--n_skip",type=int,default=4)
    parser.add_argument("--n_re",type=int,default=5)
    parser.add_argument("--out_csv",type=str,required=True)
    args=parser.parse_args()

    f1_key=args.f1_key if args.f1_key is not None else find_dataset_key(args.h5_path)
    data=load_h5_dataset(args.h5_path,f1_key)
    f1_grid=reshape_f1_to_grid(data,args.n_qp,args.n_skip,args.n_re)
    df=calc_sensitivity(f1_grid)

    out_dir=os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir,exist_ok=True)

    df.to_csv(args.out_csv,index=False)

if __name__=="__main__":
    main()
