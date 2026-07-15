# lagent.py
"""Lyapunov-based rule agent for configuration adaptation"""

import numpy as np
import pandas as pd
import random
from config import *

df = pd.read_csv('/home/dell/lyra/CASVA/coding_time/ILCAS/coding_time_DETRAC_ILCAS.csv')
ENCODE_T = df['mean_time'].tolist()

class LyapunovAgent:
    def __init__(self):
        self.v_parameter = LYAPUNOV_V
        self.l_max = L_MAX
        self.configurations = self._generate_configurations()

    def _generate_configurations(self):
        """Generate all possible configuration combinations"""
        configs = []
        for i, res in enumerate(VIDEO_CONFIGS['resolutions']):
            for j, fps in enumerate(VIDEO_CONFIGS['frame_rates']):
                for k, qp in enumerate(VIDEO_CONFIGS['qp_values']):
                    configs.append((i, j, k))
        return configs

    def select_configuration(self, env, Q):
        """Select configuration using Lyapunov optimization"""
        best_config = None
        best_objective = float('inf')
        for config in self.configurations:
            r, s, qp = config
            l_tran, l_inf, ap = env.get_info(qp, s, r)
            # Skip if latency is too high
            if l_tran > 5.0:
                continue
            # Calculate reward r_t = ap - (α × l_inf + β × l_tran)/g(f)
            reward = ap - (ALPHA * l_inf + BETA * l_tran) / (4-s)

            # Lyapunov objective: min q(t) · l_t - V · r_t
            objective = Q * l_tran - self.v_parameter * reward

            if objective < best_objective:
                best_objective = objective
                best_config = config

        # Fallback to lowest quality if no suitable config found
        if best_config is None:
            best_config = (3, 3, 4)

        return best_config

    def genetic_algorithm_optimization(self, env, Q, video_encoding_time, population_size=20, generations=100):
        """Use genetic algorithm for configuration optimization"""
        if Q == 0:
            return 1, 1, 1
        # Initialize population
        population = random.sample(self.configurations,min(population_size, len(self.configurations)))
        F0 = 0
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for config in population:
                r, s, qp = config
                knob = qp * 20 + s * 5 + r
                et = video_encoding_time[knob]
                l_tran, l_inf, ap = env.get_info(qp, s, r, et)
                if l_tran > 5.0:
                    fitness = -1000
                else:
                    reward = ap - (ALPHA * l_inf + BETA * l_tran) / (4-s)
                    objective = Q * l_tran - self.v_parameter * reward
                    fitness = -objective  # Negative because we want to minimize

                fitness_scores.append(fitness)

            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected = [population[i] for i in sorted_indices[:population_size // 2]]

            # Crossover and mutation
            new_population = selected.copy()
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

            F_new = max(fitness_scores)
            if F0 != 0 and abs(F_new - F0) < 0.01:
                break
            F0 = F_new

        # Return best configuration
        fitness_sc = []
        for c in population:
            r, s, qp = c
            knob = qp * 20 + s * 5 + r
            et = video_encoding_time[knob]
            l_tran, l_inf, ap = env.get_info(qp, s, r, et)
            reward = ap - (ALPHA * l_inf + BETA * l_tran) / (4-s)
            objective = Q * l_tran - self.v_parameter * reward
            fitness_sc.append(-objective)

        best_idx = np.argmax(fitness_sc)
        return population[best_idx]

    def _crossover(self, parent1, parent2):
        """Crossover operation for genetic algorithm"""
        child = tuple(parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1)))
        return child

    def _mutate(self, config, mutation_rate=0.1):
        """Mutation operation for genetic algorithm"""
        mutated = list(config)
        if random.random() < mutation_rate:
            idx = random.randint(0, 2)
            if idx == 0:
                mutated[0] = random.randint(0, len(VIDEO_CONFIGS['resolutions']) - 1)
            elif idx == 1:
                mutated[1] = random.randint(0, len(VIDEO_CONFIGS['frame_rates']) - 1)
            elif idx == 2:
                mutated[2] = random.randint(0, len(VIDEO_CONFIGS['qp_values']) - 1)
        return tuple(mutated)