import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from utils import get_seq_chunks_list_by_h5

FPS = [1.0, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]

fo = 25
# video_file = '/home/dell/lyra/CASVA/dataset/video_LMOT_sort_CASVA_train'
h5_file = '/home/ubuntu/lyra/CASVA/h5_file/desc/D²-City_desc.h5'

def get_seq_chunks_list(SEQ):
    chunk_list = sorted(os.listdir(video_file))
    seq_chunks = {seq_id: 0 for seq_id in range(SEQ)}
    for chunk in chunk_list:
        seq_id = int(chunk.split('_')[0])
        seq_chunks[seq_id] += 1
    return seq_chunks

# 定义待拟合的函数  [文中公式（2）]
def accuracy_model(X, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    r, f, q = X
    term1 = (c1 - c2 * np.exp(c3 * r))
    term2 = (c4 - c5 * np.exp(c6 * f))
    term3 = (c7 * q**2 + c8 * q + c9)
    return term1 * term2 * term3

# [文中公式（5）]
def size_model(X , c10, c11, c12, c13):
    r, f, q = X
    term1 = fo * f
    term2 = c10 * r**2 + c11 * r + c12
    term3 = np.exp((-c13) * q)
    return term1 * term2 * term3

def accuracy_fit():
    r_data = []
    f_data = []
    q_data = []
    a_data = []
    SEQ_TOTAL = 40
    SEQ_START = 51
    seq_chunks_list = get_seq_chunks_list_by_h5(h5_file)
    # seq_chunks_list = get_seq_chunks_list(SEQ_TOTAL)
    df = pd.read_hdf(h5_file, 'encoding_data')
    for seq in range(SEQ_TOTAL):
        for chunk_id in range(seq_chunks_list[SEQ_START+seq]):
            for j in range(5):  # QP:j
                for m in range(4):  # skip:m
                    for n in range(5):  # re:n
                        r_data.append(RE[n])
                        f_data.append(FPS[m])
                        q_data.append(QP[j])
                        a_data.append(df.loc[(SEQ_START+seq, chunk_id, j, m, n), 'Accuracy'])
    # 堆叠独立变量
    X_data = np.vstack((r_data, f_data, q_data))
    # 初始猜测参数
    initial_guess = [1.0, 1.0, -0.01, 1.0, 1.0, -0.01, 1.0, 1.0, 1.0]
    # 参数边界
    # lower_bounds = [0, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    # upper_bounds = [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf, np.inf]
    # 执行曲线拟合
    params_opt, params_covariance = curve_fit(accuracy_model, X_data, a_data, p0=initial_guess, maxfev=100000)
    # 获取拟合参数
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = params_opt
    print(c1, c2, c3, c4, c5, c6, c7, c8, c9)
    # print("拟合得到的参数值：")
    # print(f"c1 = {c1}")
    # print(f"c2 = {c2}")
    # print(f"c3 = {c3}")
    # print(f"c4 = {c4}")
    # print(f"c5 = {c5}")
    # print(f"c6 = {c6}")
    # print(f"c7 = {c7}")
    # print(f"c8 = {c8}")
    # print(f"c9 = {c9}")

    # 评估拟合效果
    a_pred = accuracy_model(X_data, *params_opt)
    residuals = a_data - a_pred
    mse = np.mean(residuals**2)
    print(f"均方误差：{mse}")

def size_fit():
    r_data = []
    f_data = []
    q_data = []
    s_data = []
    SEQ_TOTAL = 40
    SEQ_START = 51
    seq_chunks_list = get_seq_chunks_list_by_h5(h5_file)
    # seq_chunks_list = get_seq_chunks_list(SEQ_TOTAL)
    df = pd.read_hdf(h5_file, 'encoding_data')
    for seq in range(SEQ_TOTAL):
        for chunk_id in range(seq_chunks_list[SEQ_START+seq]):
            for j in range(5):  # QP:j
                for m in range(4):  # skip:m
                    for n in range(5):  # re:n
                        r_data.append(RE[n])
                        f_data.append(FPS[m])
                        q_data.append(QP[j])
                        s_data.append((df.loc[(SEQ_START+seq, chunk_id, j, m, n), 'Size']) / 1000000)


    # 堆叠独立变量
    X_data = np.vstack((r_data, f_data, q_data))
    # 初始猜测参数
    initial_guess = [-0.5, 2, 0.02, 0.10]
    # 参数边界
    # lower_bounds = [0, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    # upper_bounds = [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf, np.inf, np.inf]
    # 执行曲线拟合
    params_opt, params_covariance = curve_fit(size_model, X_data, s_data, p0=initial_guess, maxfev=100000)
    # 获取拟合参数
    c10, c11, c12, c13 = params_opt
    print(c10, c11, c12, c13)

    # print("拟合得到的参数值：")
    # print(f"c10 = {c10}")
    # print(f"c11 = {c11}")
    # print(f"c12 = {c12}")
    # print(f"c13 = {c13}")

    # 评估拟合效果
    s_pred = size_model(X_data, *params_opt)
    residuals = s_data - s_pred
    mse = np.mean(residuals**2)
    print(f"均方误差：{mse}")

if __name__ == '__main__':
    # accuracy_fit()
    size_fit()