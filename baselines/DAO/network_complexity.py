import torch
import time
import resource
import platform
import os
from Student import StudentActor


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
    fusion_model,
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
    # 1. 模型统一设置到同一个 device / dtype
    # ===============================
    fusion_model = fusion_model.to(device=device, dtype=dtype).eval()

    # ===============================
    # 2. 构造固定输入
    # ===============================
    # IL state shape: [B, 7, 8]
    batch_IL_states = torch.randn(
        B, 7, 8,
        device=device,
        dtype=dtype
    )

    batch_IL_states[:, 3, :] = torch.randint(
        0, 5, (B, 8),
        device=device
    ).to(dtype)

    batch_IL_states[:, 4, :] = torch.randint(
        0, 4, (B, 8),
        device=device
    ).to(dtype)

    batch_IL_states[:, 5, :] = torch.randint(
        0, 5, (B, 8),
        device=device
    ).to(dtype)

    # ===============================
    # 3. dtype / device / shape 检查
    # ===============================
    print("\n【输入检查】")
    print(f"  fusion_model dtype:     {next(fusion_model.parameters()).dtype}")
    print(f"  fusion_model device:    {next(fusion_model.parameters()).device}")

    print(f"  batch_IL_states dtype:  {batch_IL_states.dtype}")
    print(f"  batch_IL_states device: {batch_IL_states.device}")
    print(f"  batch_IL_states shape:  {tuple(batch_IL_states.shape)}")

    # ===============================
    # 4. Warm-up
    # ===============================
    with torch.no_grad():
        for _ in range(20):
            _ = fusion_model(batch_IL_states)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    cpu_times = []

    # ===============================
    # 5. 正式推理测试
    # ===============================
    with torch.no_grad():
        for _ in range(n_iters):

            if device.type == 'cuda':
                torch.cuda.synchronize()

            cpu_t0 = time.process_time()
            t0 = time.perf_counter()

            _ = fusion_model(batch_IL_states)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            cpu_t1 = time.process_time()

            latencies.append((t1 - t0) * 1000)
            cpu_times.append((cpu_t1 - cpu_t0) * 1000)

    latencies = torch.tensor(latencies)
    cpu_times = torch.tensor(cpu_times)

    # ===============================
    # 6. 输出统计结果
    # ===============================
    print("\n【推理时间】")
    print(f"  mean: {latencies.mean():.6f} ms")
    print(f"  max:  {latencies.max():.6f} ms")
    print(f"  min:  {latencies.min():.6f} ms")
    print(f"  P50:  {latencies.median():.6f} ms")
    print(f"  P95:  {latencies.quantile(0.95):.6f} ms")
    print(f"  P99:  {latencies.quantile(0.99):.6f} ms")

    print("\n【CPU】")
    print(f"  avg CPU time: {cpu_times.mean():.6f} ms")

    print("\n【吞吐量】")
    total_time = latencies.sum() / 1000
    print(f"  {n_iters / total_time:.0f} samples/sec")


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":

    MODEL_PATH = "/home/ubuntu/lyra/LCA/Results/Fusion/Student_20000.model"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # ===============================
    # 1. 内存 before
    # ===============================
    mem_before = get_current_memory_mb()
    peak_before = get_peak_memory_mb()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ===============================
    # 2. 初始化模型
    # ===============================
    fusion_actor = StudentActor()
    fusion_actor.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    # ===============================
    # 4. 统一 dtype
    # ===============================
    fusion_actor = fusion_actor.float()

    # ===============================
    # 5. 开始测试
    # ===============================
    stress_test_decision_net(
        fusion_actor,
        B=1,
        n_iters=1000,
        dtype=dtype,
        device=device
    )

    # ===============================
    # 6. 内存 after
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

    if device == "cuda":
        print("\n【GPU 显存】")
        print(
            f"  peak allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB"
        )
        print(
            f"  peak reserved:  "
            f"{torch.cuda.max_memory_reserved() / 1024 / 1024:.1f} MB"
        )