import numpy as np
from test import test
import os

MODEL = 'Results/CASVA_31900.model'

def QoVA(acc, bw_use, lag):
    diff = acc - 0.7
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
    # ------------------------------------------
    # PPO是网络结构，PPO_train执行训练
    # env是训练用的环境，env_fix是测试用的
    # main测试
    # Driving3_QP.h5包含1h的视频，共1800个视频段
    # coding_time.csv是统计的平均编码时间，不同的配置都有一个索引
    # environment.yml中的包很多都是没用的
    
    # 注：因为测试需要，测试的时候视频和带宽轨迹都cut成6min了，test_trace里是处理好的轨迹，
    #     可以按照自己的设置改
    #
    # ------------------------------------------
    # TOTAL = 86400
    # TOTAL = 14400
    # TOTAL = 1800
    TOTAL = 180
    output_dir = f'Results/4G/Driving3/CASVA'
    os.makedirs(output_dir, exist_ok=True)
    # 将整个视频分成10小段，每小段180*2s=360s=6min
    for k in range(10):
        # 400条带宽轨迹
        for j in range(400):
            F1, lag, bw_use, reward, bw_name, lag_1, lag_2, lag_3, lag_4, lag_5 = test(MODEL, j, 0, 1800, TOTAL, k*180)
            with open(f'{output_dir}/{k + 1:02d}_{bw_name}', 'w') as f:
                for i in range(len(F1)):
                    f.write(f"{F1[i]} {lag[i]} {bw_use[i]} {lag_1[i]} {lag_2[i]} {lag_3[i]} {lag_4[i]} {lag_5[i]}\n")
        print(k)



