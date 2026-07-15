import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *

S_INFO = 7
S_LEN = 8  # past 8
A_DIM = 100

class StudentActor(nn.Module):
    """Lightweight student network with TCN blocks"""

    def __init__(self):
        super(StudentActor, self).__init__()

        # TCN blocks for temporal modeling
        self.tcn1 = TemporalConvBlock(S_INFO, TCN_CHANNELS, TCN_KERNEL_SIZE)
        self.tcn2 = TemporalConvBlock(TCN_CHANNELS, 64, TCN_KERNEL_SIZE)

        # Feature extraction
        self.fc = nn.Linear(64, A_DIM)

    def forward(self, state):
        """
        Forward pass
        Args:
            state: Input tensor (batch, features, seq_len)
        Returns:
            policy_logits: Action logits (batch, num_actions)
        """
        # TCN processing
        x = self.tcn1(state)
        x = self.tcn2(x)

        # Global pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Feature extraction
        logits = self.fc(x)
        actor = F.softmax(logits, dim=1)

        return actor, logits, x

class StudentCritic(nn.Module):
    """Lightweight student network with TCN blocks"""

    def __init__(self):
        super(StudentCritic, self).__init__()

        # TCN blocks for temporal modeling
        self.tcn1 = TemporalConvBlock(S_INFO, TCN_CHANNELS, TCN_KERNEL_SIZE)
        self.tcn2 = TemporalConvBlock(TCN_CHANNELS, 64, TCN_KERNEL_SIZE)

        # Feature extraction
        self.fc = nn.Linear(64, 1)

    def forward(self, state):
        """
        Forward pass
        Args:
            state: Input tensor (batch, features, seq_len)
        Returns:
            policy_logits: Action logits (batch, num_actions)
        """
        # TCN processing
        x = self.tcn1(state)
        x = self.tcn2(x)

        # Global pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Feature extraction
        critic = self.fc(x)

        return critic

class TemporalConvBlock(nn.Module):
    """TCN block for temporal modeling"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(TemporalConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.residual is not None:
            residual = self.residual(residual)

        return F.relu(out + residual)