import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S_LEN = 8  # past 8
A_DIM = 100
# H4 = math.ceil(720 / 4)
# W4 = math.ceil(1280 / 4)

class MotionFeatureCNN(nn.Module):
    def __init__(self, motion_dim=384):
        super().__init__()
        self.motion_dim = motion_dim

        # --- 原论文结构 ---
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        # --- 自动推导 flatten 后维度 ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 36, 36)
            flat_dim = self.features(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, self.motion_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # 不要在网络内部写死 cuda / half
        # 统一对齐到当前模块参数的 device 和 dtype
        param = next(self.parameters())
        x = x.to(device=param.device, dtype=param.dtype, non_blocking=True)

        # 外部输入保持 [B, 36, 36]
        # Conv2d 内部需要 [B, 1, 36, 36]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.fc(x)

        return x

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.motion_dim = 384
        self.channel_fc = 128

        # —— 1. 6 条 1-D CNN 分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.channel_fc, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten()
            )
            for _ in range(6)
        ])

        # —— 2. 非 motion 特征全连接降维
        in_dim = 6 * self.channel_fc * (S_LEN - 3) + 1
        self.fc_state = nn.Linear(in_dim, self.channel_fc)

        # —— 3. motion 特征分支
        self.mfcnn = MotionFeatureCNN(motion_dim=self.motion_dim)

        # —— 4. 融合后输出动作概率
        self.fc_merge = nn.Linear(self.channel_fc + self.motion_dim, self.channel_fc)
        self.policy = nn.Linear(self.channel_fc, A_DIM)

    def forward(self, s, motion_map):
        # 统一输入到当前模型参数的 device 和 dtype
        param = next(self.parameters())
        s = s.to(device=param.device, dtype=param.dtype, non_blocking=True)
        motion_map = motion_map.to(device=param.device, dtype=param.dtype, non_blocking=True)

        # — 1) 处理 6 条历史序列
        feats = []
        for i, branch in enumerate(self.branches):
            x = s[:, i:i + 1, :]          # [B, 1, S_LEN]
            feats.append(branch(x))      # [B, 128 * (S_LEN - 3)]

        # — 2) 取最新一帧 buffer
        buf = s[:, 6:7, -1]              # [B, 1]

        seq_feat = torch.cat(feats + [buf], dim=1)
        seq_feat = F.relu(self.fc_state(seq_feat))  # [B, 128]

        # — 3) 处理 motion
        if motion_map.dim() == 3:
            # 输入为 [B, 36, 36]
            # MotionFeatureCNN 内部自动变成 [B, 1, 36, 36]
            mf_feat = self.mfcnn(motion_map)
        else:
            # 输入已经是 [B, motion_dim]
            mf_feat = motion_map

        # — 4) 融合 & 输出
        x = torch.cat([seq_feat, mf_feat], dim=1)
        x = F.relu(self.fc_merge(x))

        return F.softmax(self.policy(x), dim=-1)
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_dim = 384
        self.channel_fc = 128
        # —— 1. 6 条 1-D CNN 分支 (volume, throughput, delay, r, f, q 共 6 条)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.channel_fc, kernel_size=4, stride=1, padding=0),  # 输入长度 hist_len → 输出长度 hist_len-3
                nn.ReLU(),
                nn.Flatten()  # 每条输出维度 = 128 * (hist_len-3)
            )
            for _ in range(6)
        ])
        # —— 2. 非 motion 特征先全连接降维到 128
        in_dim = 6 * self.channel_fc * (S_LEN - 3) + 1
        self.fc_state = nn.Linear(in_dim, self.channel_fc)

        # —— 3. motion 特征分支（假设已经生成 motion_map 大小合适）
        self.mfcnn = MotionFeatureCNN(motion_dim=self.motion_dim)
        # self.mfcnn = torch.compile(self.mfcnn, mode="reduce-overhead", fullgraph=True, dynamic=False)
        # self.mfcnn = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=3),
        #     nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=3),
        #     nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(32, self.motion_dim),  # 映射到 384
        #     nn.ReLU()
        # )

        # —— 4. 拼接后再一层全连接，再输出动作
        self.fc_merge = nn.Linear(self.channel_fc + self.motion_dim, self.channel_fc)
        self.policy = nn.Linear(self.channel_fc, 1)

    def forward(self, s, motion_map):
        # — 1) 处理 6 条历史序列
        feats = []
        for i, branch in enumerate(self.branches):
            x = s[:, i:i+1, :]  # (B,1,S_LEN)
            feats.append(branch(x))  # (B,128*(S_LEN-3))
            # — 2) 取最新一帧 buffer
        buf = s[:, 6:7, -1]
        seq_feat = torch.cat(feats + [buf], dim=1)  # (B, seq_dim)
        seq_feat = F.relu(self.fc_state(seq_feat))  # (B,128)

        # — 3) 处理 motion
        mf_map = motion_map.unsqueeze(1) # (B, 1, H4, W4)
        mf_feat = self.mfcnn(mf_map)

        # — 4) 融合 & 输出
        x = torch.cat([seq_feat, mf_feat], dim=1)  # (B,128+motion_dim)
        x = F.relu(self.fc_merge(x))  # (B,128)
        return self.policy(x)  # (B, 1)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_dim = 384
        self.channel_fc = 128
        # —— 1. 6 条 1-D CNN 分支 (volume, throughput, delay, r, f, q 共 6 条)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.channel_fc, kernel_size=4, stride=1, padding=0),  # 输入长度 hist_len → 输出长度 hist_len-3
                nn.ReLU(),
                nn.Flatten()  # 每条输出维度 = 128 * (hist_len-3)
            )
            for _ in range(6)
        ])
        # —— 2. 非 motion 特征先全连接降维到 128
        in_dim = 6 * self.channel_fc * (S_LEN - 3) + 1
        self.fc_state = nn.Linear(in_dim, self.channel_fc)

        # —— 3. motion 特征分支（假设已经生成 motion_map 大小合适）
        self.mfcnn = MotionFeatureCNN(motion_dim=self.motion_dim)
        # self.mfcnn = nn.Sequential(
        #     nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=3),
        #     nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=3),
        #     nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(32, self.motion_dim),  # 映射到 384
        #     nn.ReLU()
        # )

        # —— 4. 拼接后再一层全连接，再输出动作
        self.fc_merge = nn.Linear(self.channel_fc + self.motion_dim, self.channel_fc)
        # 4) 判别头：拼接动作 one-hot → 1 + Sigmoid
        self.head = nn.Sequential(
            nn.Linear(128 + A_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, s: torch.Tensor, motion_map: torch.Tensor, a: torch.LongTensor):
        """
        state: Tensor (B, S_INFO, S_LEN)
        a:     Tensor (B,) 动作索引
        """
        B = a.size(0)
        # — 1) 处理 6 条历史序列
        feats = []
        for i, branch in enumerate(self.branches):
            x = s[:, i:i+1, :]  # (B,1,S_LEN)
            feats.append(branch(x))  # (B,128*(S_LEN-3))
            # — 2) 取最新一帧 buffer
        buf = s[:, 6:7, -1]
        seq_feat = torch.cat(feats + [buf], dim=1)  # (B, seq_dim)
        seq_feat = F.relu(self.fc_state(seq_feat))  # (B,128)

        # # — 3) 处理 motion
        # mf_map = motion_map.unsqueeze(1) # (B, 1, H4, W4)
        # mf_feat = self.mfcnn(mf_map)
        # # print(mf_feat.shape)

        # motion 部分
        if motion_map.dim() == 3:  # [B, 36, 36]
            mf_map = motion_map.unsqueeze(1)
            mf_feat = self.mfcnn(mf_map)
        else:  # [B, motion_dim]
            mf_feat = motion_map

        # — 4) 融合 & 输出
        x = torch.cat([seq_feat, mf_feat], dim=1)  # (B,128+motion_dim)
        x = F.relu(self.fc_merge(x))  # (B,128)

        # ——— 5) one-hot 动作 ———
        a_oh = torch.zeros(B, self.head[0].in_features - x.size(1), device=a.device, dtype=x.dtype) # (B,num_cfg)
        a_oh.scatter_(1, a.unsqueeze(1), 1.0)     # (B,num_cfg)

        # ——— 6) 判别输出 ———
        x = torch.cat([x, a_oh], dim=1)        # (B,128+num_cfg)
        return self.head(x).squeeze(-1)           # (B,)
