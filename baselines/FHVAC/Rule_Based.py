import random
import numpy as np
from utils import load_one_trace
import scipy.stats as stats
import os

L = 2
Lmax = 0.5 # 容许的最大延迟
V = 1 # 重视 QoI 效益（准确率、推理速度）的程度
l1 = 3.5 # 公式10的lamda1
l2 = 0.5 # 公式10的lamda2
I_max = 100 # 最大迭代次数
t_end = 0.01 # 目标值收敛到0.01时停止迭代，这里是指通过优化函数（即公式 10 中的目标函数）计算得到的结果，通常为视频配置选择带来的综合效益，比如推理准确率、延迟、带宽利用等。
fo = [25, 20, 20, 25]
INFER_T = 0.0025

FPS = [1.0, 0.5, 0.3333, 0.2]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
RE = [1, 0.64, 0.36, 0.25, 0.16]

# A是QP的可取值索引，B是FPS和RE可取值索引
A = [0, 1, 2, 3, 4]
B = [0, 1, 2, 3]

# 替换成用其他数据拟合的参数
# D²-City_720p
acc_param = [[1.3896413142477662, 0.2659519747257579, -6.213120253094989, 0.9742944931143692, 1.0144077941433867,
             -6.994206570457862, -0.10999724753324883, 0.24852239514397115, 0.5915536829452702],
             [1.3657012499637302, 0.35036413713134673, -4.560719846839296, 1.2004317521981942, 0.5268910630072101,
              -5.6326916583182465, -0.19285324777705612, 0.37588891112270756, 0.32286846898340926],
             [1.0088071259900124, 0.9895148716503459, -0.0006827921770303651, 1.2884343940886018, 0.5635530444566971,
              -4.839338956667894, -6.07119622397089, 15.646020864609284, 18.73486736078288],
             [1.3071446820027388, 0.35941818469791914, -5.968924996501108, 1.3777691081440957, 0.30922529074907557,
             -5.29571875647406, -0.11039066190805628, 0.2519757523585613, 0.30270232561785293],
             ]
size_param = [[0.5110841270488347, 0.2920437522018229, 0.12963633144357523, 3.45836625475459],
              [0.3104713224159204, 1.2423263607115314, 0.15613943874320102, 3.0629175315407595],
              [11.640502181931392, 2.7729834296412696, -0.08725643881338849, 0.2802758775001444],
              [-0.3761504299164407, 1.19043997099841, -0.00834819631597256, 2.778395765519282]]


def initialize_population(size):
    population = []
    for _ in range(size):
        r = random.choice(A)
        f = random.choice(B)
        q = random.choice(A)
        population.append((int(r), int(f), int(q)))
    return population

def calculate_fitness(individual, Q, bw, idx):
    r, f, q = individual
    target_value = target(Q, RE[r], FPS[f], QP[q], bw, idx)
    # if target_value <= 0:
    #     return float('inf')
    # return 1.0 / target_value
    return -target_value

def tournament_selection(population, fitness, k=3):
    selected = random.sample(population, k)
    best_individual = max(selected, key=lambda x: fitness[x])
    return best_individual

def two_point_crossover(parent1, parent2):
    r1, f1, q1 = parent1
    r2, f2, q2 = parent2
    child1 = (r1, f2, q1)
    child2 = (r2, f1, q2)
    return child1, child2

def mutation(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        # r, f, q
        return random.choice(A), random.choice(B), random.choice(A)
    return individual

def accuracy_model(r, f, q, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    term1 = (c1 - c2 * np.exp(c3 * r))
    term2 = (c4 - c5 * np.exp(c6 * f))
    # term3 = (c7 - c8 * np.exp(c9 * f))
    term3 = (c7 * q**2 + c8 * q + c9)
    return term1 * term2 * term3

def latency_model(r, f, q, bw, c10, c11, c12, c13, idx):
    term1 = fo[idx] * f
    term2 = c10 * r**2 + c11 * r + c12
    term3 = np.exp((-c13) * q)
    s = term1 * term2 * term3 * 1000000
    return s/(bw*1000)

# 公式(10)
def target(Q, r, f, q, bw, idx):
    a_t = accuracy_model(r, f, q, *acc_param[idx])
    p_t = fo[idx] * f * L * INFER_T
    l_t = latency_model(r, f, q, bw, *size_param[idx], idx)
    Q = max(Q+l_t-Lmax, 0)
    return Q * l_t - V * (a_t  + l1 * p_t +l2 * l_t)

def rule_based(Q, bw, idx):
    if Q == 0 and bw == 0:
        return 1, 1, 1
    N = 20
    # 初始化种群
    population = initialize_population(N)

    fitness = {ind: calculate_fitness(ind, Q, bw, idx) for ind in population}
    F0 = max(fitness.values())
    i = 0
    while i <= I_max:
        best_individual = max(fitness, key=fitness.get)
        # 让种群中一半的个体都是最优个体
        stud_pop = [best_individual] * (N // 2)
        rest_pop = [random.choice(population) for _ in range(N - len(stud_pop))]
        temp_pop = [tournament_selection(population, fitness) for _ in range(len(rest_pop))]
        new_population = stud_pop + temp_pop
        offspring = []
        for j in range(0, len(new_population), 2):
            if j + 1 < len(new_population):
                parent1, parent2 = new_population[j], new_population[j + 1]
                child1, child2 = two_point_crossover(parent1, parent2)
                offspring.append(mutation(child1))
                offspring.append(mutation(child2))
        population = offspring
        fitness = {ind: calculate_fitness(ind, Q, bw, idx) for ind in population}
        F_new = max(fitness.values())
        if i > I_max or abs(F_new - F0) < t_end:
            break
        F0 = F_new
        i += 1
    selected_config = best_individual
    return selected_config

def test(index, schunks_list):
    cooked_bw, cooked_name = load_one_trace('/home/dell/lyra/CASVA/train_trace/', index)
    seq_total = 11
    seq_start = 0
    f1_sum = 0.0
    lag_sum = 0.0
    reward_sum = 0.0
    for seq_id in range(seq_total):
        env = env_fix.Environment(cooked_bw=cooked_bw, start=0, seq_chunks=schunks_list[seq_start+seq_id], seq_id=seq_total + seq_id)
        # env = env_fix.Environment(cooked_bw=cooked_bw, start=start, chunk=chunk, total=total, chunk_start=chunk_start)
        qp = 1
        skip = 1
        re = 1
        end_of_video = False
        while not end_of_video:
            # 可以看出规则策略就是根据过往带宽和长尾上传延迟来决定选择什么样的配置
            bw_est, _, _, _, _, _, _, Q, end_of_video = env.get_video_chunk(qp, skip, re)
            best = rule_based(Q, bw_est, idx)
            qp = best[2]
            skip = best[1]
            re = best[0]

        f1_mean = np.mean(env.F1)
        # f1_std = np.std(env.F1, ddof=1)
        # f1_standard_error = f1_std / np.sqrt(len(env.F1))
        # f1_interval = stats.norm.interval(0.95, loc=f1_mean, scale=f1_standard_error)

        lag_mean = np.mean(env.lag)
        # lag_std = np.std(env.lag, ddof=1)
        # lag_standard_error = lag_std / np.sqrt(len(env.lag))
        # lag_interval = stats.norm.interval(0.95, loc=lag_mean, scale=lag_standard_error)

        reward_mean = np.mean(env.Reward)

        print(seq_id,"：",f1_mean, lag_mean, reward_mean)


        f1_sum += f1_mean
        lag_sum += lag_mean
        reward_sum += reward_mean

    print("==================================================")
    f1_avg = f1_sum / seq_total
    lag_avg = lag_sum / seq_total
    reward_avg = reward_sum / seq_total
    print("average result：",f1_avg, lag_avg, reward_avg)


if __name__ == '__main__':
    seq_chunks_list = get_seq_chunks_list(11)
    test(0, seq_chunks_list)
