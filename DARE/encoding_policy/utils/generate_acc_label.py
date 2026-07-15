import pandas as pd
import numpy as np

MAX_DROP = 0.02

df = pd.read_hdf("/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/D²-City_desc.h5", "encoding_data").reset_index()
label_dict = {}
grouped = df.groupby(["SEQ", "CHUNK"])

def enforce_monotonic(acc_list):
    corrected = []
    min_so_far = acc_list[0]

    for acc in acc_list:
        min_so_far = min(min_so_far, acc)
        corrected.append(min_so_far)

    return corrected

for (video, block), group in grouped:
    baseline = group[
        (group["QP"] == 0) &
        (group["SKIP"] == 0) &
        (group["RE"] == 0)
    ]
    if len(baseline) == 0:
        continue
    baseline_acc = baseline["Accuracy"].values[0]
    qp_candidates = sorted(group["QP"].unique())
    acc_values = []
    for i in qp_candidates:
        rows = group[
            (group["QP"] == i) &
            (group["SKIP"] == 0) &
            (group["RE"] == 0)
        ]
        if len(rows) == 0:
            continue
        acc_values.append(rows["Accuracy"].values[0])
        # acc = rows["Acc"].values[0]
        # if baseline_acc - acc <= MAX_DROP or acc >= baseline_acc:
        #     qp_idx = i
        # else:
        #     break
    acc_values = enforce_monotonic(acc_values)
    qp_idx = 0
    for i, acc in enumerate(acc_values):
        if baseline_acc - acc <= MAX_DROP:
            qp_idx = i
        else:
            break
    acc_values = []
    skip_candidates = sorted(group["SKIP"].unique())
    for i in skip_candidates:
        rows = group[
            (group["QP"] == 0) &
            (group["SKIP"] == i) &
            (group["RE"] == 0)
        ]
        if len(rows) == 0:
            continue
        acc_values.append(rows["Accuracy"].values[0])
        # acc = rows["Acc"].values[0]
        # if baseline_acc - acc <= MAX_DROP or acc >= baseline_acc:
        #     skip_idx = i
        # else:
        #     break
    acc_values = enforce_monotonic(acc_values)
    skip_idx = 0
    for i, acc in enumerate(acc_values):
        if baseline_acc - acc <= MAX_DROP:
            skip_idx = i
        else:
            break
    acc_values = []
    re_candidates = sorted(group["RE"].unique())
    for i in re_candidates:
        rows = group[
            (group["QP"] == 0) &
            (group["SKIP"] == 0) &
            (group["RE"] == i)
        ]
        if len(rows) == 0:
            continue
        acc_values.append(rows["Accuracy"].values[0])
        # acc = rows["Acc"].values[0]
        # if baseline_acc - acc <= MAX_DROP or acc >= baseline_acc:
        #     re_idx = i
        # else:
        #     break
    acc_values = enforce_monotonic(acc_values)
    re_idx = 0
    for i, acc in enumerate(acc_values):
        if baseline_acc - acc <= MAX_DROP:
            re_idx = i
        else:
            break
    label_dict[(video, block)] = {
        "qp_idx": int(qp_idx),
        "skip_idx": int(skip_idx),
        "re_idx": int(re_idx)
    }


qp_values = [v["qp_idx"] for v in label_dict.values()]
print(pd.Series(qp_values).value_counts())