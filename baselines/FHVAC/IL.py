import torch
import torch.nn as nn
import torch.nn.functional as F

S_INFO = 7
S_LEN = 8  # past 8
A_DIM = 100


class ILAgent(nn.Module):
    def __init__(self):
        super(ILAgent, self).__init__()
        # GRU layers for time-varying inputs (bw_t, l_t)
        self.gru_bw = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        self.gru_lt = nn.GRU(input_size=1, hidden_size=32, batch_first=True)

        self.fc_knob = nn.Linear(3, 16)
        self.fc_buff = nn.Linear(1, 16)
        self.fc_lq = nn.Linear(1, 16)
        # 原来没考虑pt（推理时间）
        # self.fc_lp = nn.Linear(1, 16)

        self.fc1 = nn.Linear(32 + 32 + 16 * 3, 128)
        # 加上pt后
        # self.fc1 = nn.Linear(32 + 32 + 16 * 4, 128)
        self.fc2 = nn.Linear(128, A_DIM)

    def forward(self, inputs):
        # bw_batch = inputs[:, 0:1, :]
        # delay_batch = inputs[:, 1:2, :]
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)  # Reshape to (batch, seq, feature)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)

        buff_batch = inputs[:, 2:3, :]
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]
        l_q_batch= inputs[:, 6:7, :]
        # 加上pt后
        # l_p_batch = inputs[:, 7:8, :]
        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]
        buff = buff_batch[:, :, -1]
        l_q = l_q_batch[:, :, -1]
        # l_p = l_p_batch[:, :, -1]

        x_1 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))
        x_2 = F.relu(self.fc_buff(buff))
        x_3 = F.relu(self.fc_lq(l_q))
        # 加上pt后
        # x_6 = F.relu(self.fc_lp(l_p))

        _, x_4 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_5 = self.gru_lt(delay_batch)

        x = torch.cat([x_4.squeeze(0), x_5.squeeze(0), x_1, x_2, x_3], dim=1)
        # 加上pt后
        # x = torch.cat([x_4.squeeze(0), x_5.squeeze(0), x_1, x_2, x_3, x_6], dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        output = F.softmax(self.fc2(x), dim=1)
        return output, logits

    def get_feature(self, inputs):
        # bw_batch = inputs[:, 0:1, :]
        # delay_batch = inputs[:, 1:2, :]
        bw_batch = inputs[:, 0:1, :].permute(0, 2, 1)  # Reshape to (batch, seq, feature)
        delay_batch = inputs[:, 1:2, :].permute(0, 2, 1)

        buff_batch = inputs[:, 2:3, :]
        qp_batch = inputs[:, 3:4, :]
        skip_batch = inputs[:, 4:5, :]
        re_batch = inputs[:, 5:6, :]
        l_q_batch = inputs[:, 6:7, :]
        # 加上pt后
        # l_p_batch = inputs[:, 7:8, :]
        qp = qp_batch[:, :, -1]
        skip = skip_batch[:, :, -1]
        re = re_batch[:, :, -1]
        buff = buff_batch[:, :, -1]
        l_q = l_q_batch[:, :, -1]
        # l_p = l_p_batch[:, :, -1]

        x_1 = F.relu(self.fc_knob(torch.cat([re, skip, qp], dim=1)))
        x_2 = F.relu(self.fc_buff(buff))
        x_3 = F.relu(self.fc_lq(l_q))
        # 加上pt后
        # x_6 = F.relu(self.fc_lp(l_p))

        _, x_4 = self.gru_bw(bw_batch)  # GRU output for bandwidth
        _, x_5 = self.gru_lt(delay_batch)

        features = torch.cat([x_4.squeeze(0), x_5.squeeze(0), x_1, x_2, x_3], dim=1)
        # 加上pt后
        # features = torch.cat([x_4.squeeze(0), x_5.squeeze(0), x_1, x_2, x_3, x_6], dim=1)
        return features