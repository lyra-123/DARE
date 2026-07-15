import numpy as np

RTT = 0.08  # 80ms
class ExpertDP:
    def __init__(self, L=2, T=2):
        self.L, self.T = L, T

    def solve(self, acc, ud):
        n, c = acc.shape
        step = 0.1
        s = int(self.L / step) + 1
        A = np.full((n + 1, s), -np.inf, dtype=float)
        S = np.full((n + 1, s), -1, int)
        Q = np.full((n + 1, s, 2), -1, int)
        A[0, 0] = 0.0
        for i in range(n):
            for j in range(s):
                if A[i, j] < 0: continue
                lag = j * step
                for k in range(c):
                    lag2 = max(0.0, lag + ud[i, k] - self.T)
                    if lag2 > self.L: lag2 = self.L
                    idx = int(round(lag2 / step))
                    # 边界保护
                    if idx < 0:
                        idx = 0
                    elif idx >= s:
                        idx = s - 1
                    v = A[i, j] + acc[i, k]
                    if v > A[i + 1, idx]:
                        A[i + 1, idx] = v
                        S[i + 1, idx] = k
                        Q[i + 1, idx] = (i, j)

            # 保底检查：如果下一层仍然全不可达，则从当前层选择最优 prev_j 强制推进（避免中断）
            if not np.any(np.isfinite(A[i + 1, :])):
                # 选择当前层累计值最大的状态 prev_j
                prev_j = int(np.nanargmax(A[i, :]))  # nanargmax 在全 -inf 时也会报错，但 A[0,0] 初始为 0 保证 i=0 有可达
                # 从 prev_j 强制尝试所有动作（不会跳过）
                lag = prev_j * step
                for k in range(c):
                    lag2 = max(0.0, lag + ud[i, k] - self.T)
                    if lag2 > self.L:
                        lag2 = self.L
                    idx = int(round(lag2 / step))
                    idx = min(max(idx, 0), s - 1)
                    v = A[i, prev_j] + acc[i, k]
                    if v > A[i + 1, idx]:
                        A[i + 1, idx] = v
                        S[i + 1, idx] = int(k)
                        Q[i + 1, idx] = (int(i), int(prev_j))

        # j = A[n].argmax()
        # path = []
        # i = n
        # while i > 0:
        #     k = S[i, j]
        #     path.append(k)
        #     i, j = Q[i, j]

        j = int(np.nanargmax(A[n, :]))
        # 回溯得到 path（并做安全保护）
        path = []
        i = n
        while i > 0:
            k = int(S[i, j]) if S[i, j] >= 0 else -1
            if k == -1:
                # 出现未记录动作的极端情况，用贪心补齐：选择该步 acc 最大的动作
                # 注意：此时 i>0，对应 acc[i-1]
                greedy_k = int(np.argmax(acc[i - 1]))
                path.append(greedy_k)
                # 试着向上步进一格（若 Q 无效，手动回退 i）
                prev = Q[i, j]
                if prev[0] < 0:
                    i -= 1
                    # j 保持不变或重置为 0
                    j = 0
                else:
                    i, j = int(prev[0]), int(prev[1])
                continue

            path.append(k)
            prev = Q[i, j]
            # 如果前驱无效（-1），直接把剩余步骤补为贪心动作并终止回溯循环
            if prev[0] < 0:
                # 补齐剩余 i-1 .. 0 步（倒序加入）
                for ii in range(i - 1, 0, -1):
                    greedy_k = int(np.argmax(acc[ii - 1]))
                    path.append(greedy_k)
                break
            i, j = int(prev[0]), int(prev[1])

        path = list(reversed(path))

        # 最终保证长度为 n：若不足则用贪心动作补齐；若超出（不应出现）则截断
        if len(path) < n:
            # 从已有 path 末尾或直接从 acc 补全
            for ii in range(len(path), n):
                greedy_k = int(np.argmax(acc[ii]))
                path.append(greedy_k)
        elif len(path) > n:
            path = path[-n:]
        return path
