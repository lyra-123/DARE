import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

S_INFO = 7  # bw, delay, buffer, qp, skip, r, segment_size, dynamics
S_LEN = 8  # past 8
A_DIM = 100

# Define the Actor network
class RLActor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(RLActor, self).__init__()
        self.gru_bw = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)
        self.gru_lt = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)
        self.gru_size = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)

        self.fc_knob = nn.Linear(3, PPO_HIDDEN_SIZE)

        self.features = ResNet18_pretrained()
        self.feat_fc = nn.Linear(512, PPO_HIDDEN_SIZE)

        self.fc1 = nn.Linear(PPO_HIDDEN_SIZE * 4, PPO_HIDDEN_SIZE)
        self.fc2 = nn.Linear(PPO_HIDDEN_SIZE, A_DIM)

        self.expand_dropout = nn.Dropout(p=0.2)

        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable=True):
        """动态设置特征提取器是否可训练"""
        for param in self.features.parameters():
            param.requires_grad = trainable
        self.freeze_backbone = not trainable

    def forward(self, inputs, flow_feat):
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)
        size_batch = inputs[:, 2:3, :].permute(0, 2, 1)
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]

        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]

        _, x_1 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_2 = self.gru_lt(delay_batch)
        _, x_3 = self.gru_size(size_batch)

        x_4 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))

        seq_feat = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4], dim=1)
        seq_feat = F.relu(self.fc1(seq_feat))

        if isinstance(flow_feat, torch.Tensor) and flow_feat.dim() == 2:  # 预计算特征 (B, 128)
            pass
        else:  # 原始图像 (B, T, C, H, W)
            if self.freeze_backbone:
                with torch.no_grad():
                    flow_feat = self.extract_features(flow_feat)
            else:
                flow_feat = self.extract_features(flow_feat)
        flow_feat = F.relu(self.feat_fc(flow_feat))

        # STBlock
        outer = torch.einsum('bd,be->bde', seq_feat, flow_feat)
        outer = self.expand_dropout(outer)
        squeezed = outer.sum(dim=2)
        signed_sqrt = torch.sign(squeezed) * torch.sqrt(torch.abs(squeezed) + 1e-10)
        l2_norm = F.normalize(signed_sqrt, p=2, dim=1)

        logits = self.fc2(l2_norm)
        actor = F.softmax(logits, dim=1)
        return actor

    def extract_features(self, flow_feat):
        flow_feat = flow_feat.permute(0, 3, 1, 2)
        flow_feat = self.features(flow_feat)
        return flow_feat

    def get_feature(self, inputs, flow_feat):
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)
        size_batch = inputs[:, 2:3, :].permute(0, 2, 1)
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]

        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]

        _, x_1 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_2 = self.gru_lt(delay_batch)
        _, x_3 = self.gru_size(size_batch)

        x_4 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))

        seq_feat = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4], dim=1)
        seq_feat = F.relu(self.fc1(seq_feat))

        if isinstance(flow_feat, torch.Tensor) and flow_feat.dim() == 2:  # 预计算特征 (B, 128)
            pass
        else:  # 原始图像 (B, T, C, H, W)
            if self.freeze_backbone:
                with torch.no_grad():
                    flow_feat = self.extract_features(flow_feat)
            else:
                flow_feat = self.extract_features(flow_feat)
        flow_feat = F.relu(self.feat_fc(flow_feat))

        early_feat = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4, flow_feat], dim=1)

        # STBlock
        outer = torch.einsum('bd,be->bde', seq_feat, flow_feat)
        outer = self.expand_dropout(outer)
        squeezed = outer.sum(dim=2)
        signed_sqrt = torch.sign(squeezed) * torch.sqrt(torch.abs(squeezed) + 1e-10)
        late_feat = F.normalize(signed_sqrt, p=2, dim=1)

        logits = self.fc2(late_feat)

        return early_feat, late_feat, logits

# Define the Critic network
class RLCritic(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(RLCritic, self).__init__()
        self.gru_bw = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)
        self.gru_lt = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)
        self.gru_size = nn.GRU(input_size=1, hidden_size=PPO_HIDDEN_SIZE, batch_first=True)

        self.fc_knob = nn.Linear(3, PPO_HIDDEN_SIZE)

        self.features = ResNet18_pretrained()
        self.feat_fc = nn.Linear(512, PPO_HIDDEN_SIZE)

        self.fc1 = nn.Linear(PPO_HIDDEN_SIZE * 4, PPO_HIDDEN_SIZE)
        self.fc2 = nn.Linear(PPO_HIDDEN_SIZE, 1)

        self.expand_dropout = nn.Dropout(p=0.2)

        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable=True):
        """动态设置特征提取器是否可训练"""
        for param in self.features.parameters():
            param.requires_grad = trainable
        self.freeze_backbone = not trainable

    def forward(self, inputs, flow_feat):
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)
        size_batch = inputs[:, 2:3, :].permute(0, 2, 1)
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]

        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]

        _, x_1 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_2 = self.gru_lt(delay_batch)
        _, x_3 = self.gru_size(size_batch)

        x_4 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))

        seq_feat = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4], dim=1)
        seq_feat = F.relu(self.fc1(seq_feat))

        if isinstance(flow_feat, torch.Tensor) and flow_feat.dim() == 2:  # 预计算特征 (B, 128)
            pass
        else:  # 原始图像 (B, T, C, H, W)
            if self.freeze_backbone:
                with torch.no_grad():
                    flow_feat = self.extract_features(flow_feat)
            else:
                flow_feat = self.extract_features(flow_feat)
        flow_feat = F.relu(self.feat_fc(flow_feat))

        # STBlock
        outer = torch.einsum('bd,be->bde', seq_feat, flow_feat)
        outer = self.expand_dropout(outer)
        squeezed = outer.sum(dim=2)
        signed_sqrt = torch.sign(squeezed) * torch.sqrt(torch.abs(squeezed) + 1e-10)
        l2_norm = F.normalize(signed_sqrt, p=2, dim=1)

        critic = self.fc2(l2_norm)
        return critic
    
    def extract_features(self, flow_feat):
        flow_feat = flow_feat.permute(0, 3, 1, 2)
        flow_feat = self.features(flow_feat)
        return flow_feat

def ResNet18_pretrained(pretrained=True):
    import torchvision.models as models
    new_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    new_model.avgpool = nn.AdaptiveMaxPool2d((1, 1))
    new_model.fc = nn.Identity()
    return new_model


