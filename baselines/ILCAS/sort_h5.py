# import pandas as pd
#
# # 读取 HDF5 文件
# df = pd.read_hdf('/home/dell/lyra/ILCAS/h5_file/sample_cpa/desc/LMOT_desc.h5', 'encoding_data').reset_index()
#
# # 显示修改前的部分数据（可选）
# print("Before modification:")
# print(df.head())
#
# # 使用replace方法同时修改多个SEQ值
# df['SEQ'] = df['SEQ'].replace({5:8,8:5})
#
# # 显示修改后的数据（可选）
# print("\nAfter modification:")
# print(df.head())
#
# # 保存修改后的 DataFrame 到新的 HDF5 文件
# df.to_hdf('/home/dell/lyra/ILCAS/h5_file/sample_cpa/sort/LMOT_sort.h5', 'encoding_data', mode='w')
#
# # 输出为文本文件
# with open('/home/dell/lyra/ILCAS/h5_file/sample_cpa/sort/output.txt', 'w', encoding='utf-8') as f:
#     f.write(df.to_string())
#
# print("Data has been modified and saved successfully.")

import h5py


def swap_vid_str_and_save(h5_path, output_h5_path, seq_replacements):
    """
    根据给定的映射交换视频组数据，只交换指定的 `vid_str`，并保存为新文件。

    Parameters:
        h5_path (str): 输入的 HDF5 文件路径。
        output_h5_path (str): 输出的 HDF5 文件路径。
        seq_replacements (dict): 要替换的 seq_id 映射字典，用于交换对应编号的数据。
    """
    # 读取原始 HDF5 文件
    with h5py.File(h5_path, 'r') as f:
        # 创建一个新的 HDF5 文件用于保存修改后的数据
        with h5py.File(output_h5_path, 'w') as new_f:
            # 遍历每个视频组
            video_ids = sorted(f.keys(), key=lambda x: int(x))
            # 记录需要交换的 seq_id
            seq_to_swap = list(seq_replacements.keys())

            # 先复制所有没有修改的组
            for vid_str in video_ids:
                # 如果这个组既不是键也不是值，就直接复制
                if int(vid_str) not in seq_to_swap and int(vid_str) not in seq_replacements.values():
                    # 直接复制没有修改的组
                    new_group = new_f.create_group(vid_str)
                    new_group.create_dataset('features', data=f[vid_str]['features'][:])

            # 交换需要修改的 seq_id
            for old_seq, new_seq in seq_replacements.items():
                print(old_seq, new_seq)
                if str(old_seq) in f and str(new_seq) in f:
                    # 交换对应的数据
                    feats_old = f[str(old_seq)]['features'][:]  # (num_chunks, C)
                    feats_new = f[str(new_seq)]['features'][:]

                    # 创建新组并交换数据
                    new_group_old = new_f.create_group(str(new_seq))
                    new_group_old.create_dataset('features', data=feats_old)  # 将 old_seq 的数据放到 new_seq
                    new_group_new = new_f.create_group(str(old_seq))
                    new_group_new.create_dataset('features', data=feats_new)  # 将 new_seq 的数据放到 old_seq

            print(f"Data has been swapped and saved successfully to {output_h5_path}.")


# 示例调用
h5_path = '/home/dell/lyra/Can/deg_feats/layer1/LMOT_layer1.h5'
output_h5_path = '/home/dell/lyra/Can/deg_feats/sort/LMOT_layer1.h5'

# 定义需要交换的 seq_id 映射
seq_replacements = {
    5:8
}


# 调用函数进行交换并保存
swap_vid_str_and_save(h5_path, output_h5_path, seq_replacements)