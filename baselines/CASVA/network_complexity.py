import torch
import time
import resource
import platform
import os
from PPO import Actor


# ===============================
# 内存函数
# ===============================
def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == 'Darwin':
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def get_current_memory_mb():
    status_path = '/proc/self/status'
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    return get_peak_memory_mb()


# ===============================
# 决策网络复杂度测试
# ===============================
def stress_test_decision_net(
    model,
    B=1,
    n_iters=1000,
    dtype=torch.float32,
    device='cpu'
):
    print("\n" + "=" * 60)
    print(f"Decision Network 推理测试: {n_iters} 次")
    print("=" * 60)

    device = torch.device(device)

    # ===============================
    # 1. 模型设置
    # ===============================
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # ===============================
    # 2. 构造固定输入
    # ===============================
    # state shape: [B, 7, 8]
    batch_states = torch.randn(
        B, 8, 8,
        device=device,
        dtype=dtype
    )

    # 第 3、4、5 行分别对应 QP / SKIP / RE 等离散动作历史
    # 注意：randint 默认生成 int64，这里转成 float32，与网络输入保持一致
    batch_states[:, 3, :] = torch.randint(
        0, 5, (B, 8),
        device=device
    ).to(dtype)

    batch_states[:, 4, :] = torch.randint(
        0, 4, (B, 8),
        device=device
    ).to(dtype)

    batch_states[:, 5, :] = torch.randint(
        0, 5, (B, 8),
        device=device
    ).to(dtype)

    # motion_map shape: [B, 36, 36]
    # 当 B=1 时，shape 就是 [1, 36, 36]
    batch_motion_map = torch.randn(
        B, 36, 36,
        device=device,
        dtype=dtype
    )

    # ===============================
    # 3. dtype / shape 检查
    # ===============================
    print("\n【输入检查】")
    print(f"  model dtype:        {next(model.parameters()).dtype}")
    print(f"  batch_states dtype: {batch_states.dtype}")
    print(f"  motion_map dtype:   {batch_motion_map.dtype}")
    print(f"  batch_states shape: {tuple(batch_states.shape)}")

    # ===============================
    # 4. Warm-up
    # ===============================
    with torch.no_grad():
        for _ in range(20):
            _ = model(
                batch_states.float(),
            )

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()



    latencies = []
    cpu_times = []

    # ===============================
    # 6. 正式推理测试
    # ===============================
    with torch.no_grad():
        for _ in range(n_iters):

            if device.type == 'cuda':
                torch.cuda.synchronize()

            cpu_t0 = time.process_time()
            t0 = time.perf_counter()

            prob = model(
                batch_states.float(),
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            cpu_t1 = time.process_time()

            latencies.append((t1 - t0) * 1000)
            cpu_times.append((cpu_t1 - cpu_t0) * 1000)

    latencies = torch.tensor(latencies)
    cpu_times = torch.tensor(cpu_times)

    # ===============================
    # 8. 输出统计结果
    # ===============================
    print("\n【推理时间】")
    print(f"  mean: {latencies.mean():.6f} ms")
    print(f"  max: {latencies.max():.6f} ms")
    print(f"  min: {latencies.min():.6f} ms")
    print(f"  P50:  {latencies.median():.6f} ms")
    print(f"  P95:  {latencies.quantile(0.95):.6f} ms")
    print(f"  P99:  {latencies.quantile(0.99):.6f} ms")

    print("\n【CPU】")
    print(f"  avg CPU time: {cpu_times.mean():.6f} ms")



# ===============================
# 主函数
# ===============================
if __name__ == "__main__":

    MODEL_PATH = "/home/ubuntu/lyra/CASVA/Results/CASVA_3100_0.630261_2.359271.model"
    # ===============================
    # 5. 内存 before
    # ===============================
    mem_before = get_current_memory_mb()
    peak_before = get_peak_memory_mb()

    # 1. 初始化模型
    model_actor = Actor()

    # 2. 加载参数
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_actor.load_state_dict(state_dict)

    # 3. 强制模型为 float32
    model_actor = model_actor.float()

    # 4. 开始测试
    stress_test_decision_net(
        model_actor,
        B=1,
        n_iters=1000,
        dtype=torch.float32,
        device='cuda'      # 如果测试 GPU，改成 'cuda'
    )

    # ===============================
    # 7. 内存 after
    # ===============================
    mem_after = get_current_memory_mb()
    peak_after = get_peak_memory_mb()

    print("\n【空间复杂度（进程法）】")
    print(
        f"  current: {mem_before:.1f} MB → {mem_after:.1f} MB "
        f"(+{mem_after - mem_before:.1f} MB)"
    )
    print(
        f"  peak:    {peak_before:.1f} MB → {peak_after:.1f} MB "
        f"(+{peak_after - peak_before:.1f} MB)"
    )

    print("\n【GPU 显存】")
    print(
        f"  peak allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB"
    )
    print(
        f"  peak reserved:  "
        f"{torch.cuda.max_memory_reserved() / 1024 / 1024:.1f} MB"
    )