import os
import h5py
import argparse
import numpy as np
import pandas as pd


def find_dataset_key(h5_path, candidates=None):
    if candidates is None:
        candidates = ["f1", "accuracy", "acc", "score"]

    keys = []

    with h5py.File(h5_path, "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)
        f.visititems(collect)

    for key in keys:
        low = key.lower()
        for cand in candidates:
            if cand.lower() in low:
                return key

    raise KeyError("F1 dataset key not found.")


def load_h5_dataset(h5_path, key):
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[key], dtype=np.float32)


def reshape_f1_to_grid(data, n_qp, n_skip, n_re):
    total = n_qp * n_skip * n_re

    if data.ndim >= 3 and tuple(data.shape[-3:]) == (n_qp, n_skip, n_re):
        return data.reshape(-1, n_qp, n_skip, n_re)

    if data.shape[-1] == total:
        return data.reshape(-1, total).reshape(-1, n_qp, n_skip, n_re)

    raise ValueError(f"Invalid shape: {data.shape}")


def calc_sensitivity_one(x):
    qp_curve = np.mean(x, axis=(1, 2))
    skip_curve = np.mean(x, axis=(0, 2))
    re_curve = np.mean(x, axis=(0, 1))

    v_qp = np.var(qp_curve, ddof=0)
    v_skip = np.var(skip_curve, ddof=0)
    v_re = np.var(re_curve, ddof=0)

    s_qp = 2.0 * np.sqrt(v_qp)
    s_skip = 2.0 * np.sqrt(v_skip)
    s_re = 2.0 * np.sqrt(v_re)

    return s_qp, s_skip, s_re, v_qp, v_skip, v_re


def calc_sensitivity(f1_grid, insensitive_thr):
    rows = []
    names = ["QP", "SKIP", "RE"]

    for i in range(f1_grid.shape[0]):
        s_qp, s_skip, s_re, v_qp, v_skip, v_re = calc_sensitivity_one(f1_grid[i])

        sens = np.array([s_qp, s_skip, s_re], dtype=np.float32)
        var = np.array([v_qp, v_skip, v_re], dtype=np.float32)

        dominant_id = int(np.argmax(sens))
        dominant_param = names[dominant_id]
        max_sens = float(np.max(sens))

        total_sens = float(np.sum(sens))
        total_var = float(np.sum(var))

        label = "insensitive" if max_sens < insensitive_thr else dominant_param

        rows.append({
            "sample_id": i,
            "s_qp": float(s_qp),
            "s_skip": float(s_skip),
            "s_re": float(s_re),
            "v_qp": float(v_qp),
            "v_skip": float(v_skip),
            "v_re": float(v_re),
            "s_qp_ratio": float(s_qp / total_sens) if total_sens > 1e-12 else 0.0,
            "s_skip_ratio": float(s_skip / total_sens) if total_sens > 1e-12 else 0.0,
            "s_re_ratio": float(s_re / total_sens) if total_sens > 1e-12 else 0.0,
            "v_qp_ratio": float(v_qp / total_var) if total_var > 1e-12 else 0.0,
            "v_skip_ratio": float(v_skip / total_var) if total_var > 1e-12 else 0.0,
            "v_re_ratio": float(v_re / total_var) if total_var > 1e-12 else 0.0,
            "max_sensitivity": max_sens,
            "dominant_param": dominant_param,
            "label": label,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--f1_key", type=str, default=None)
    parser.add_argument("--n_qp", type=int, default=5)
    parser.add_argument("--n_skip", type=int, default=4)
    parser.add_argument("--n_re", type=int, default=5)
    parser.add_argument("--insensitive_thr", type=float, default=0.02)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    f1_key = args.f1_key if args.f1_key is not None else find_dataset_key(args.h5_path)

    data = load_h5_dataset(args.h5_path, f1_key)
    f1_grid = reshape_f1_to_grid(data, args.n_qp, args.n_skip, args.n_re)
    df = calc_sensitivity(f1_grid, args.insensitive_thr)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()