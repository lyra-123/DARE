# -*- coding: utf-8 -*-

import os
import warnings
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy.stats import mode as sp_mode
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
import time

warnings.filterwarnings("ignore")
try:
    import tables  # noqa: F401
except ImportError:
    raise ImportError(
        "缺少 pandas.read_hdf() 所需依赖 PyTables。\n"
        "请先安装：\n"
        "  conda install -c conda-forge pytables\n"
        "或：\n"
        "  python -m pip install tables"
    )

ENCODING_H5 = {
    "DETRAC":  "/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DETRAC_desc.h5",
    "DSEC":    "/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/DSEC_desc.h5",
    "LMOT":    "/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/LMOT_desc.h5",
    "D²-City": "/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/D²-City_desc.h5",
}

FEATURE_H5 = {
    "DETRAC":  "../deg_feats/layer1/layer1/DETRAC_layer1.h5",
    "DSEC":    "../deg_feats/layer1/layer1/DSEC_layer1.h5",
    "LMOT":    "../deg_feats/layer1/layer1/LMOT_layer1.h5",
    "D²-City": "../deg_feats/layer1/layer1/D²-City_layer1.h5",
}

DATASET_NAMES = ["DETRAC", "DSEC", "LMOT", "D²-City"]

SAVE_DIR = "results/labels_to_cluster/2"


def build_labels_from_encoding_h5(encoding_h5_dict, dataset_names):
    """
    对每个 (SEQ, CHUNK)，从所有编码配置中选：
        F1 最高；
        如果 F1 相同，则 Size 最小；
    的配置作为标签。

    标签为：
        (QP, SKIP, RE)
    """
    label_dict = {}

    for ds_name in dataset_names:
        h5_path = encoding_h5_dict[ds_name]
        print(f"[Label] Loading {ds_name} from {h5_path} ...")

        df = pd.read_hdf(h5_path, "encoding_data")
        df.columns = [c.strip().upper() for c in df.columns]

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if ds_name == "DETRAC":
            df = df[(df["SEQ"] >= 0) & (df["SEQ"] <= 13)]

        if ds_name == "D²-City":
            df = df[(df["SEQ"] >= 51) & (df["SEQ"] <= 105)]

        col_map = {}
        for c in df.columns:
            if "SEQ" in c:
                col_map[c] = "SEQ"
            elif "CHUNK" in c:
                col_map[c] = "CHUNK"
            elif "QP" in c:
                col_map[c] = "QP"
            elif "SKIP" in c:
                col_map[c] = "SKIP"
            elif "RE" in c:
                col_map[c] = "RE"
            elif "SIZE" in c:
                col_map[c] = "SIZE"
            elif "ACC" in c or "F1" in c or "ACCURACY" in c:
                col_map[c] = "F1"
            elif "BITRATE" in c or "BIT" in c:
                col_map[c] = "BITRATE"

        df = df.rename(columns=col_map)

        required = {"SEQ", "CHUNK", "QP", "SKIP", "RE", "SIZE", "F1"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{ds_name}: 缺少列 {missing}，现有列: {list(df.columns)}"
            )

        ds_labels = {}

        for (seq, chunk), group in df.groupby(["SEQ", "CHUNK"]):
            best = group.sort_values(
                by=["F1", "SIZE"],
                ascending=[False, True],
            ).iloc[0]

            ds_labels[(int(seq), int(chunk))] = (
                int(best["QP"]),
                int(best["SKIP"]),
                int(best["RE"]),
            )

        label_dict[ds_name] = ds_labels
        print(f"  {ds_name}: {len(ds_labels)} 个 (seq, chunk) 标签")

    return label_dict

def load_features(feature_h5_dict, dataset_names):
    feat_dict = {}

    for ds_name in dataset_names:
        h5_path = feature_h5_dict[ds_name]
        print(f"[Feature] Loading {ds_name} from {h5_path} ...")

        ds_feats = {}

        with h5py.File(h5_path, "r") as f:
            for vid in sorted(f.keys(), key=lambda x: int(x)):
                feats = f[vid]["features"][:]

                for chunk_idx, feat in enumerate(feats):
                    if feat.ndim == 4:
                        p1_map = feat[0]
                        diff_map = feat[2]
                    else:
                        p1_map = feat
                        diff_map = feat

                    p1_vec = p1_map.mean(axis=(-1, -2))
                    diff_vec = diff_map.mean(axis=(-1, -2))

                    ds_feats[(int(vid), int(chunk_idx))] = (p1_vec, diff_vec)

        feat_dict[ds_name] = ds_feats
        print(f"  {ds_name}: {len(ds_feats)} 个 (video_id, chunk_idx) 特征")

    return feat_dict

def align_labels_and_features(label_dict, feat_dict, dataset_names):
    p1_vecs = []
    diff_vecs = []

    qp_labels = []
    skip_labels = []
    re_labels = []

    meta = []
    skipped = 0

    for ds_idx, ds_name in enumerate(dataset_names):
        labels = label_dict[ds_name]
        feats = feat_dict[ds_name]

        matched = 0

        for (seq, chunk), (qi, si, ri) in sorted(labels.items()):
            key = (int(seq), int(chunk))

            if key not in feats:
                skipped += 1
                continue

            p1_vec, diff_vec = feats[key]

            p1_vecs.append(p1_vec)
            diff_vecs.append(diff_vec)

            qp_labels.append(qi)
            skip_labels.append(si)
            re_labels.append(ri)

            meta.append({
                "ds_idx": int(ds_idx),
                "ds_name": ds_name,
                "video_id": int(seq),
                "chunk_idx": int(chunk),
                "global_idx": int(len(meta)),
            })

            matched += 1

        print(f"  {ds_name}: {matched} 条对齐，累计跳过 {skipped} 条")

    p1_vecs = np.array(p1_vecs, dtype=np.float32)
    diff_vecs = np.array(diff_vecs, dtype=np.float32)

    qp_labels = np.array(qp_labels, dtype=np.int32)
    skip_labels = np.array(skip_labels, dtype=np.int32)
    re_labels = np.array(re_labels, dtype=np.int32)

    print(f"\n对齐完成: {len(meta)} 条块级样本，跳过 {skipped} 条")

    return p1_vecs, diff_vecs, qp_labels, skip_labels, re_labels, meta

def aggregate_to_sequence_level(
    p1_vecs,
    diff_vecs,
    qp_labels,
    skip_labels,
    re_labels,
    meta,
):
    groups = defaultdict(list)

    for m in meta:
        groups[(m["ds_name"], m["video_id"])].append(m["global_idx"])

    seq_p1 = []
    seq_diff = []

    seq_qp = []
    seq_skip = []
    seq_re = []

    seq_meta = []

    for (ds_name, video_id), indices in sorted(groups.items()):
        seq_idx = len(seq_meta)
        idxs = np.array(indices)

        p1s = p1_vecs[idxs]
        diffs = diff_vecs[idxs]

        qps = qp_labels[idxs]
        skips = skip_labels[idxs]
        res = re_labels[idxs]

        seq_p1.append(p1s.mean(axis=0))
        seq_diff.append(diffs.mean(axis=0))

        seq_qp.append(int(sp_mode(qps, keepdims=True).mode[0]))
        seq_skip.append(int(sp_mode(skips, keepdims=True).mode[0]))
        seq_re.append(int(sp_mode(res, keepdims=True).mode[0]))

        ds_idx = meta[indices[0]]["ds_idx"]

        seq_meta.append({
            "ds_idx": int(ds_idx),
            "ds_name": ds_name,
            "video_id": int(video_id),
            "n_chunks": int(len(indices)),
            "seq_idx": int(seq_idx),
        })

    seq_p1 = np.array(seq_p1, dtype=np.float32)
    seq_diff = np.array(seq_diff, dtype=np.float32)

    seq_qp = np.array(seq_qp, dtype=np.int32)
    seq_skip = np.array(seq_skip, dtype=np.int32)
    seq_re = np.array(seq_re, dtype=np.int32)

    print(f"\n序列级聚合完成: {len(seq_meta)} 条序列")

    return seq_p1, seq_diff, seq_qp, seq_skip, seq_re, seq_meta


def label_guided_tree_clustering(
    p1_vecs,
    diff_vecs,
    qp_labels,
    skip_labels,
    re_labels,
    max_leaves=7,
    min_samples_leaf=5,
    random_state=42,
):
    scaler = StandardScaler()

    X_raw = np.concatenate([p1_vecs, diff_vecs], axis=1)
    X_scaled = scaler.fit_transform(X_raw)

    C = p1_vecs.shape[1]

    feature_names = (
        [f"p1_ch{i}" for i in range(C)] +
        [f"diff_ch{i}" for i in range(C)]
    )

    Y = np.stack([qp_labels, skip_labels, re_labels], axis=1)

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaves,
        min_samples_leaf=min_samples_leaf,
        criterion="gini",
        random_state=random_state,
    )

    tree.fit(X_scaled, Y)

    internal_leaf_ids = tree.apply(X_scaled)
    unique_internal_leaves = np.unique(internal_leaf_ids)

    remap = {
        int(old_leaf_id): int(new_leaf_id)
        for new_leaf_id, old_leaf_id in enumerate(unique_internal_leaves)
    }

    leaf_labels = np.array(
        [remap[int(lid)] for lid in internal_leaf_ids],
        dtype=np.int32,
    )

    counts = dict(zip(*np.unique(leaf_labels, return_counts=True)))

    print("\n决策树训练完成")
    print(f"  max_leaves       = {max_leaves}")
    print(f"  min_samples_leaf = {min_samples_leaf}")
    print(f"  actual_leaves    = {len(unique_internal_leaves)}")
    print(f"  leaf counts      = {counts}")

    print("\n标准化尺度下的树结构，仅用于检查：")
    print(export_text(tree, feature_names=feature_names, max_depth=20))

    return leaf_labels, tree, scaler, remap, feature_names

def threshold_to_raw_scale(threshold_z, scaler, feature_id):
    """
    StandardScaler:
        z = (raw - mean) / std

    因此：
        raw = z * std + mean
    """
    return float(threshold_z * scaler.scale_[feature_id] + scaler.mean_[feature_id])


def save_leaf_path_feature_values(
    tree,
    scaler,
    remap,
    feature_names,
    seq_p1,
    seq_diff,
    seq_meta,
    seq_leaf_labels,
    seq_qp,
    seq_skip,
    seq_re,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    t = tree.tree_

    X_raw = np.concatenate([seq_p1, seq_diff], axis=1)

    X_scaled = scaler.transform(X_raw)

    internal_leaf_ids = tree.apply(X_scaled)
    leaf_paths = {}

    def recurse(node_id, conditions):
        feature_id = int(t.feature[node_id])

        # 到达叶子节点
        if feature_id == _tree.TREE_UNDEFINED:
            internal_leaf_id = int(node_id)
            external_leaf_id = int(remap.get(internal_leaf_id, internal_leaf_id))

            leaf_paths[internal_leaf_id] = {
                "leaf_id": external_leaf_id,
                "internal_leaf_id": internal_leaf_id,
                "conditions": conditions,
            }
            return

        feature_name = feature_names[feature_id]

        threshold_z = float(t.threshold[node_id])
        threshold_raw = threshold_to_raw_scale(threshold_z, scaler, feature_id)

        left_child = int(t.children_left[node_id])
        right_child = int(t.children_right[node_id])

        recurse(
            left_child,
            conditions + [{
                "node_id": int(node_id),
                "feature_id": int(feature_id),
                "feature_name": feature_name,
                "operator": "<=",
                "threshold_original": threshold_raw,
                "threshold_standardized": threshold_z,
                "condition_text": f"{feature_name} <= {threshold_raw:.8f}",
            }],
        )

        recurse(
            right_child,
            conditions + [{
                "node_id": int(node_id),
                "feature_id": int(feature_id),
                "feature_name": feature_name,
                "operator": ">",
                "threshold_original": threshold_raw,
                "threshold_standardized": threshold_z,
                "condition_text": f"{feature_name} > {threshold_raw:.8f}",
            }],
        )

    recurse(0, [])

    if len(leaf_paths) == 0:
        raise RuntimeError("没有找到任何叶子节点路径，请检查决策树是否训练成功。")

    max_depth = max(len(v["conditions"]) for v in leaf_paths.values())

    rows = []

    for internal_leaf_id, path_info in leaf_paths.items():
        leaf_id = int(path_info["leaf_id"])
        conditions = path_info["conditions"]

        path_text = " AND ".join([c["condition_text"] for c in conditions])

        sample_indices = np.where(internal_leaf_ids == internal_leaf_id)[0]

        for si in sample_indices:
            m = seq_meta[si]

            row = {
                "leaf_id": leaf_id,
                "internal_leaf_id": int(internal_leaf_id),

                "seq_idx": int(m["seq_idx"]),
                "ds_idx": int(m["ds_idx"]),
                "ds_name": m["ds_name"],
                "video_id": int(m["video_id"]),
                "n_chunks": int(m["n_chunks"]),

                "seq_qp_label": int(seq_qp[si]),
                "seq_skip_label": int(seq_skip[si]),
                "seq_re_label": int(seq_re[si]),

                "path_depth": int(len(conditions)),
                "path_text": path_text,
            }
            for j, cond in enumerate(conditions, start=1):
                fid = int(cond["feature_id"])

                value = float(X_raw[si, fid])
                threshold = float(cond["threshold_original"])
                operator = cond["operator"]

                if operator == "<=":
                    satisfied = value <= threshold
                else:
                    satisfied = value > threshold

                row[f"cond{j}_node_id"] = int(cond["node_id"])
                row[f"cond{j}_feature"] = cond["feature_name"]
                row[f"cond{j}_operator"] = operator
                row[f"cond{j}_threshold"] = threshold
                row[f"cond{j}_value"] = value
                row[f"cond{j}_distance"] = value - threshold
                row[f"cond{j}_satisfied"] = bool(satisfied)

            for j in range(len(conditions) + 1, max_depth + 1):
                row[f"cond{j}_node_id"] = ""
                row[f"cond{j}_feature"] = ""
                row[f"cond{j}_operator"] = ""
                row[f"cond{j}_threshold"] = ""
                row[f"cond{j}_value"] = ""
                row[f"cond{j}_distance"] = ""
                row[f"cond{j}_satisfied"] = ""

            rows.append(row)

    out_csv = os.path.join(save_dir, "leaf_path_feature_values.csv")

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["leaf_id", "ds_name", "video_id", "seq_idx"],
        ascending=[True, True, True, True],
    )

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


    return out_csv


def run(
    max_leaves=7,
    min_samples_leaf=5,
    save_dir=SAVE_DIR,
):
    os.makedirs(save_dir, exist_ok=True)

    label_dict = build_labels_from_encoding_h5(
        ENCODING_H5,
        DATASET_NAMES,
    )

    feat_dict = load_features(
        FEATURE_H5,
        DATASET_NAMES,
    )

    (
        p1_vecs,
        diff_vecs,
        qp_labels,
        skip_labels,
        re_labels,
        meta,
    ) = align_labels_and_features(
        label_dict,
        feat_dict,
        DATASET_NAMES,
    )

    (
        seq_p1,
        seq_diff,
        seq_qp,
        seq_skip,
        seq_re,
        seq_meta,
    ) = aggregate_to_sequence_level(
        p1_vecs,
        diff_vecs,
        qp_labels,
        skip_labels,
        re_labels,
        meta,
    )

    (
        seq_leaf,
        tree,
        scaler,
        remap,
        feature_names,
    ) = label_guided_tree_clustering(
        seq_p1,
        seq_diff,
        seq_qp,
        seq_skip,
        seq_re,
        max_leaves=max_leaves,
        min_samples_leaf=min_samples_leaf,
    )

    out_csv = save_leaf_path_feature_values(
        tree=tree,
        scaler=scaler,
        remap=remap,
        feature_names=feature_names,
        seq_p1=seq_p1,
        seq_diff=seq_diff,
        seq_meta=seq_meta,
        seq_leaf_labels=seq_leaf,
        seq_qp=seq_qp,
        seq_skip=seq_skip,
        seq_re=seq_re,
        save_dir=save_dir,
    )


    return {
        "out_csv": out_csv,
        "seq_leaf": seq_leaf,
        "tree": tree,
        "scaler": scaler,
        "remap": remap,
        "feature_names": feature_names,
        "seq_meta": seq_meta,
    }

if __name__ == "__main__":
    train_start = time.perf_counter()
    run(
        max_leaves=7,
        min_samples_leaf=5,
        save_dir=SAVE_DIR,
    )
    train_end = time.perf_counter()
    total_train_time = train_end - train_start
    print(f"Total training time: {total_train_time:.2f} s")