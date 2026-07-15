import h5py
import pandas as pd
import cv2
import numpy as np
import torch


# class FlowMagLoader:
#     def __init__(self, flow_h5_path, mag_h5_path, actor, device='cuda'):
#         # self.flows = {}
#         self.flow_feats = {}
#         actor.eval()
#         with h5py.File(flow_h5_path, "r") as f:
#             for seq_id in f.keys():
#                 for chunk_id in f[seq_id].keys():
#                     # 直接读伪RGB，保持为 numpy (H,W,3)，uint8
#                     flow_arr = f[f"{seq_id}/{chunk_id}"][:]  # (H, W, 3), uint8
#                     # self.flows[(int(seq_id), int(chunk_id))] = flow_arr
#                     # self.flows[(int(seq_id), int(chunk_id))] = f[f"{seq_id}/{chunk_id}"][:]
#                     # --- 提取特征 ---
#                     resized = self.resize_image(flow_arr, (224, 224))
#                     flow_tensor = torch.from_numpy(resized.astype("float32") / 255.0). float()
#                     flow_tensor = flow_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
#                     with torch.no_grad():
#                         feat = actor.extract_features(flow_tensor)
#                     self.flow_feats[(int(seq_id), int(chunk_id))] = feat.cpu().numpy()
#
#         self.mags = pd.read_hdf(mag_h5_path, key="mag").set_index(["seq_id", "chunk_id"])
#
#     def get_flow_feat(self, seq_id, chunk_id):
#         """直接获取提前提取好的 flow feature"""
#         return self.flow_feats[(seq_id, chunk_id)]
#
#     def get_flow(self, seq_id, chunk_id, normalize=True, resize_to=(224,224)):
#         arr = self.flows[(seq_id, chunk_id)]  # (H, W, 3), uint8
#         if resize_to is not None:
#             arr = self.resize_image(arr, resize_to)
#         if normalize:
#             return arr.astype("float32") / 255.0  # 转 [0,1] 浮点
#         return arr
#
#     def get_mag(self, seq_id, chunk_id):
#         return float(self.mags.loc[(seq_id, chunk_id), "mag_value"])
#
#     def resize_image(self, image, size):
#         ih, iw, _ = image.shape
#         h, w = size
#         scale = min(w / iw, h / ih)
#         nw = int(iw * scale)
#         nh = int(ih * scale)
#         image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
#         image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
#         image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
#         return image_back

class MultiFlowMagLoader:
    def __init__(self, flow_h5_paths, mag_h5_paths, actor, device='cuda'):
        """
        flow_h5_paths: list[str]
        mag_h5_paths: list[str]
        """
        self.flows = {}
        self.flow_feats = {}
        self.mags = []
        self.seq_offset = 0  # 用于统一编号
        self.seq_range_map = {}  # 记录每个数据集的seq范围
        actor.eval()

        # === 遍历每个数据集 ===
        for flow_path, mag_path in zip(flow_h5_paths, mag_h5_paths):
            dataset_name = flow_path.split('/')[-1].replace('.h5', '')

            with h5py.File(flow_path, "r") as f:
                max_seq = max(int(seq_id) for seq_id in f.keys())
                for seq_id in f.keys():
                    for chunk_id in f[seq_id].keys():
                        global_seq_id = int(seq_id) + self.seq_offset
                        flow_arr = f[f"{seq_id}/{chunk_id}"][:]  # (H, W, 3), uint8
                        self.flows[(global_seq_id, int(chunk_id))] = f[f"{seq_id}/{chunk_id}"][:]

                        # --- 提取特征 ---
                        resized = self.resize_image(flow_arr, (224, 224))
                        flow_tensor = torch.from_numpy(resized.astype("float32") / 255.0).float()
                        flow_tensor = flow_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
                        with torch.no_grad():
                            feat = actor.extract_features(flow_tensor)
                        self.flow_feats[(global_seq_id, int(chunk_id))] = feat.cpu().numpy()

            mag_df = pd.read_hdf(mag_path, key="mag").reset_index()
            mag_df["seq_id"] = mag_df["seq_id"] + self.seq_offset
            self.mags.append(mag_df)

            self.seq_range_map[dataset_name] = (self.seq_offset, self.seq_offset + max_seq)
            self.seq_offset += max_seq + 1

        # 合并所有幅值信息
        self.mags = pd.concat(self.mags).set_index(["seq_id", "chunk_id"])

    def get_flow_feat(self, seq_id, chunk_id):
        """直接获取提前提取好的 flow feature"""
        return self.flow_feats[(seq_id, chunk_id)]

    def get_flow(self, seq_id, chunk_id, normalize=True, resize_to=(224, 224)):
        arr = self.flows[(seq_id, chunk_id)]  # (H, W, 3), uint8
        if resize_to is not None:
            arr = self.resize_image(arr, resize_to)
        if normalize:
            arr = arr.astype("float32") / 255.0
        return arr

    def get_mag(self, seq_id, chunk_id):
        return float(self.mags.loc[(seq_id, chunk_id), "mag_value"])

    def resize_image(self, image, size):
        ih, iw, _ = image.shape
        h, w = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        image_back[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image_resized
        return image_back
