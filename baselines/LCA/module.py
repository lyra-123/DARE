import torch
import torch.nn as nn
import torch.nn.functional as F

A_DIM = 100

class WeightAdapter(nn.Module):
    """
    Adaptive weight allocator that dynamically assigns weights to teachers
    Based on the paper's architecture: Logits -> Conv -> Average Pooling -> FC -> Softmax
    """

    def __init__(self, num_teachers=2, logits_dim=A_DIM):
        super(WeightAdapter, self).__init__()
        self.num_teachers = num_teachers
        self.logits_dim = logits_dim

        # Convolutional layer to process concatenated logits
        # Treat logits as 1D "image" with multiple channels (one per teacher)
        self.conv = nn.Conv1d(
            in_channels=num_teachers,
            out_channels=32,  # Number of conv filters
            kernel_size=3,
            padding=1
        )

        # Average pooling to reduce dimensionality
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(32, 64)

        # Final softmax layer to output weights
        self.softmax_layer = nn.Sequential(
            nn.Linear(64, num_teachers),
            nn.Softmax(dim=-1)
        )

    def forward(self, teacher_logits):
        """
        Compute adaptive weights for teachers based on their logit predictions

        Args:
            teacher_logits: List of logit tensors from teachers [batch_size, num_actions]

        Returns:
            weights: Adaptive weights for each teacher [batch_size, num_teachers]
        """
        # Stack teacher logits: [batch_size, num_teachers, num_actions]
        stacked_logits = torch.stack(teacher_logits, dim=1)

        # Apply convolution along the action dimension
        # Input: [batch_size, num_teachers, num_actions]
        conv_out = self.conv(stacked_logits)  # [batch_size, 32, num_actions]

        # Apply average pooling
        pooled = self.avg_pool(conv_out)  # [batch_size, 32, 1]
        pooled = pooled.squeeze(-1)  # [batch_size, 32]

        # Fully connected processing
        fc_out = torch.relu(self.fc(pooled))  # [batch_size, 64]

        # Generate weights with softmax
        weights = self.softmax_layer(fc_out)  # [batch_size, num_teachers]

        return weights


class ResidualBlockFusion(nn.Module):
    """
    Residual Block Fusion (RBF) module
    Fuses first and last feature layers of teacher networks
    """

    def __init__(self, early_dim, late_dim, hidden_dim=128, reduction=4):
        super(ResidualBlockFusion, self).__init__()

        # 投影层，把输入对齐到同一维度
        self.early_proj = nn.Linear(early_dim, hidden_dim)
        self.late_proj = nn.Linear(late_dim, hidden_dim)

        # FC 计算权重
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.fc2 = nn.Linear(hidden_dim // reduction, 2)  # 输出两个权重 [a, b]

    def forward(self, early_features, late_features):
        """
        Fuse early and late features with residual connection

        Args:
            early_features: Early layer features
            late_features: Late layer features

        Returns:
            fused_features: Fused feature representation
        """
        # Project to same dimension
        early_proj = self.early_proj(early_features)
        late_proj = self.late_proj(late_features)

        f_sum = early_proj + late_proj

        z = F.relu(self.fc1(f_sum))  # (B, fuse_dim//r)
        weights = F.softmax(self.fc2(z), dim=1)  # (B, 2)
        a, b = weights[:, 0:1], weights[:, 1:2]  # (B, 1)

        # 5. 加权融合
        fused_feat = a * early_proj + b * late_proj  # (B, fuse_dim)

        return fused_feat