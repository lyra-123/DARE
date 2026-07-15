import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

S_INFO = 7
S_LEN = 8  # past 8
A_DIM = 100


class ILAgent(nn.Module):
    def __init__(self):
        super(ILAgent, self).__init__()
        # GRU layers for time-varying inputs (bw_t, l_t)
        self.gru_bw = nn.GRU(input_size=1, hidden_size=IL_HIDDEN_SIZE, batch_first=True)
        self.gru_lt = nn.GRU(input_size=1, hidden_size=IL_HIDDEN_SIZE, batch_first=True)
        self.gru_lq = nn.GRU(input_size=1, hidden_size=IL_HIDDEN_SIZE, batch_first=True)

        self.fc_knob = nn.Linear(3, IL_HIDDEN_SIZE)

        self.fc = nn.Linear(IL_HIDDEN_SIZE * 4, A_DIM)

    def forward(self, inputs):
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)  # Reshape to (batch, seq, feature)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)
        l_q_batch = inputs[:, 2:3, :].permute(0, 2, 1)
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]

        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]

        _, x_1 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_2 = self.gru_lt(delay_batch)
        _, x_3 = self.gru_lq(l_q_batch)

        x_4 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))

        x = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4], dim=1)

        logits = self.fc(x)
        output = F.softmax(logits, dim=1)
        return output, logits

    def get_feature(self, inputs):
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)  # Reshape to (batch, seq, feature)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)
        l_q_batch = inputs[:, 2:3, :].permute(0, 2, 1)
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]

        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]

        _, x_1 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_2 = self.gru_lt(delay_batch)
        _, x_3 = self.gru_lq(l_q_batch)

        x_4 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))

        early_feats = torch.cat([x_1.squeeze(0), x_2.squeeze(0), x_3.squeeze(0), x_4], dim=1)
        logits = self.fc(early_feats)

        return early_feats, early_feats, logits