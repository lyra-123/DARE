# import os
# import numpy as np
#
# # 输入文件路径
# file_path = "/home/dell/lyra/CASVA/train_trace/NewFile-HighDensity-4G.txt"
#
# # 输出文件夹（自动创建）
# output_dir = "trace"
# os.makedirs(output_dir, exist_ok=True)
#
# with open(file_path, "r") as f:
#     values = [float(line.strip()) for line in f if line.strip()]
#
# segments = []
# current_segment = []
#
# # 查找连续大于25的段
# for v in values:
#     if v > 25:
#         current_segment.append(v)
#     else:
#         if len(current_segment) >= 2300:
#             segments.append(current_segment)
#         current_segment = []
#
# # 末尾段处理
# if len(current_segment) >= 2300:
#     segments.append(current_segment)
#
# print(f"共找到 {len(segments)} 个连续段（每段长度≥4000且值>25）。")
#
# # 统计文件路径
# summary_path = os.path.join(output_dir, "stats_summary.txt")
#
# # 保存每段及统计信息
# with open(summary_path, "w") as summary_file:
#     summary_file.write("filename\tmean\tvariance\tmean/variance\n")
#
#     for i, seg in enumerate(segments, 1):
#         seg = np.array(seg)
#         mean_val = np.mean(seg) / 1000
#         std_val = np.std(seg) / 1000
#         ratio = std_val /  mean_val if mean_val != 0 else float('inf')
#
#         seg_filename = f"{i:03d}.txt"
#         seg_path = os.path.join(output_dir, seg_filename)
#
#         # 保存数据段
#         with open(seg_path, "w") as f:
#             f.write("\n".join(map(str, seg)))
#         # 写入统计信息
#         summary_file.write(f"{seg_filename}\t{mean_val:.6f}\t{std_val:.6f}\t{ratio:.6f}\n")
#         print(f"已保存 {seg_filename} （长度 {len(seg)}）")
#
# print(f"\n✅ 全部完成，统计信息已保存至: {summary_path}")

import os
import numpy as np

# 输入文件路径
file_path = "/home/dell/lyra/CASVA/train_trace/NewFile-HighDensity-4G.txt"

# 输出目录
output_dir = "traces"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
with open(file_path, "r") as f:
    values = [float(line.strip()) for line in f if line.strip()]

segments = []
current_segment = []

# 查找连续不含 0 的段
for v in values:
    if v != 0:
        current_segment.append(v)
    else:
        if len(current_segment) >= 1000:
            segments.append(current_segment)
        current_segment = []

# 末尾段处理
if len(current_segment) >= 1000:
    segments.append(current_segment)

print(f"共找到 {len(segments)} 个连续段（每段长度≥1000且不含0）。")

# 统计文件路径
summary_path = os.path.join(output_dir, "stats_summary.txt")

# 保存每段及统计信息
with open(summary_path, "w") as summary_file:
    summary_file.write("filename\tmean\tvariance\tmean/variance\n")

    for i, seg in enumerate(segments, 1):
        seg = np.array(seg)
        mean_val = np.mean(seg) / 1000
        var_val = np.std(seg) / 1000
        ratio = var_val / mean_val if mean_val != 0 else float('inf')

        seg_filename = f"{i:04d}.txt"
        seg_path = os.path.join(output_dir, seg_filename)

        # 保存数据段
        with open(seg_path, "w") as f:
            f.write("\n".join(map(str, seg)))

        # 写入统计信息
        summary_file.write(f"{seg_filename}\t{mean_val:.6f}\t{var_val:.6f}\t{ratio:.6f}\n")

        print(f"已保存 {seg_filename} （长度 {len(seg)}）")

print(f"\n✅ 全部完成，统计信息已保存至: {summary_path}")