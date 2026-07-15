import numpy as np
import torch
import pandas as pd
from test import RL_test, IL_test, Fusion_test
from utils import load_one_trace
import scipy.stats as stats
import os

# RLMODEL = 'Results/history/RL.model'
# ILMODEL = 'Results/history/IL_47100.model'
# Fusion = 'Results/history/Fusion_47100.model'

RLMODEL = 'Results/RL/RL.model'
ILMODEL = 'Results/Fusion/IL_45900.model'
Fusion = 'Results/Fusion/Fusion_45900.model'

def QoVA(acc, bw_use, lag):
    diff = acc - 0.7
    # result = np.zeros_like(diff)
    # result[diff > 0] = 100 * np.log(100 * np.abs(diff[diff > 0]))
    # result[diff < 0] = -100 * np.log(100 * np.abs(diff[diff < 0]))
    if diff > 0:
        result = 100 * np.log(100 * np.abs(diff))
    elif diff < 0:
        result = -100 * np.log(100 * np.abs(diff))
    else:
        result = 0
    b = 10 * bw_use
    c = 40 * lag
    return result-b-c

if __name__ == '__main__':
    TOTAL = 180
    v_id = 0
    output_dir = f'../Results/4G/Driving3/FHVAC'
    os.makedirs(output_dir, exist_ok=True)
    for k in range(10):
        for j in range(400):
            F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = Fusion_test(RLMODEL, ILMODEL, Fusion, j, 0,
                                                           1800, TOTAL, k*180)
            with open(f'{output_dir}/{k + 1:02d}_{bw_name}', 'w') as f:
                for i in range(len(F1)):
                    f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")
        print(k)


