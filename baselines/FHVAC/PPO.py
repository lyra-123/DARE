# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# S_INFO = 8  # bw, delay, buffer, qp, skip, r, segment_size, dynamics
# S_LEN = 8  # past 8
# A_DIM = 80
#
# # Define the Actor network
# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor, self).__init__()
#         channel_cnn = 128
#         channel_fc = 128
#
#         self.bw_conv = nn.Conv1d(1, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
#         self.delay_conv = nn.Conv1d(1, channel_cnn, 4)
#         self.size_conv = nn.Conv1d(1, channel_cnn, 4)
#         self.dyn_conv = nn.Conv1d(1, channel_cnn, 4)
#
#         self.knob_fc = nn.Linear(3, channel_cnn)
#         self.buffer_fc = nn.Linear(1, channel_cnn)
#
#         incoming_size = 4 * channel_cnn * 5 + 2 * channel_cnn
#
#         # self.noise_layer = NoiseLinear(in_features=incoming_size, out_features=channel_fc)
#         self.noise_layer = nn.Linear(in_features=incoming_size, out_features=channel_fc)
#         self.fc2 = nn.Linear(in_features=channel_fc, out_features=A_DIM)
#
#
#     def forward(self, inputs):
#         bw_batch = inputs[:, 0:1, :]
#         delay_batch = inputs[:, 1:2, :]
#         buff = inputs[:, 2:3, :]
#         qp = inputs[:, 3:4, :]
#         skip = inputs[:, 4:5, :]
#         re = inputs[:, 5:6, :]
#         size_batch = inputs[:, 6:7, :]
#         dynamics_batch = inputs[:, 7:8, :]
#
#         x_1 = F.relu(self.bw_conv(bw_batch))
#         x_2 = F.relu(self.delay_conv(delay_batch))
#         x_3 = F.relu(self.size_conv(size_batch))
#         x_4 = F.relu(self.dyn_conv(dynamics_batch))
#         # x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1).squeeze(-1)))
#         qp = qp[:, :, -1]
#         skip = skip[:, :, -1]
#         re = re[:, :, -1]
#         buff = buff[:, :, -1]
#         x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1)))
#         x_6 = F.relu(self.buffer_fc(buff))
#
#         x_1 = x_1.view(x_1.size(0), -1)
#         x_2 = x_2.view(x_2.size(0), -1)
#         x_3 = x_3.view(x_3.size(0), -1)
#         x_4 = x_4.view(x_4.size(0), -1)
#
#         x = torch.cat([x_5, x_6, x_1, x_2, x_3, x_4], dim=1)
#         x = F.relu(self.noise_layer(x))
#         actor = F.softmax(self.fc2(x), dim=1)
#         return actor
#
#     def get_feature(self, inputs):
#         bw_batch = inputs[:, 0:1, :]
#         delay_batch = inputs[:, 1:2, :]
#         buff = inputs[:, 2:3, :]
#         qp = inputs[:, 3:4, :]
#         skip = inputs[:, 4:5, :]
#         re = inputs[:, 5:6, :]
#         size_batch = inputs[:, 6:7, :]
#         dynamics_batch = inputs[:, 7:8, :]
#
#         x_1 = F.relu(self.bw_conv(bw_batch))
#         x_2 = F.relu(self.delay_conv(delay_batch))
#         x_3 = F.relu(self.size_conv(size_batch))
#         x_4 = F.relu(self.dyn_conv(dynamics_batch))
#         # x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1).squeeze(-1)))
#         qp = qp[:, :, -1]
#         skip = skip[:, :, -1]
#         re = re[:, :, -1]
#         buff = buff[:, :, -1]
#         x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1)))
#         x_6 = F.relu(self.buffer_fc(buff))
#
#         x_1 = x_1.view(x_1.size(0), -1)
#         x_2 = x_2.view(x_2.size(0), -1)
#         x_3 = x_3.view(x_3.size(0), -1)
#         x_4 = x_4.view(x_4.size(0), -1)
#
#         features = torch.cat([x_5, x_6, x_1, x_2, x_3, x_4], dim=1)
#         return features
#
# # Define the Critic network
# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         channel_cnn = 128
#         channel_fc = 128
#
#         self.bw_conv = nn.Conv1d(1, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
#         self.delay_conv = nn.Conv1d(1, channel_cnn, 4)
#         self.size_conv = nn.Conv1d(1, channel_cnn, 4)
#         self.dyn_conv = nn.Conv1d(1, channel_cnn, 4)
#
#         self.knob_fc = nn.Linear(3, channel_cnn)
#         self.buffer_fc = nn.Linear(1, channel_cnn)
#
#         incoming_size = 4 * channel_cnn * 5 + 2 * channel_cnn
#         self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
#         self.fc2 = nn.Linear(in_features=channel_fc, out_features=1)
#
#     def forward(self, inputs):
#         bw_batch = inputs[:, 0:1, :]
#         delay_batch = inputs[:, 1:2, :]
#         buff = inputs[:, 2:3, :]
#         qp = inputs[:, 3:4, :]
#         skip = inputs[:, 4:5, :]
#         re = inputs[:, 5:6, :]
#         size_batch = inputs[:, 6:7, :]
#         dynamics_batch = inputs[:, 7:8, :]
#
#         x_1 = F.relu(self.bw_conv(bw_batch))
#         x_2 = F.relu(self.delay_conv(delay_batch))
#         x_3 = F.relu(self.size_conv(size_batch))
#         x_4 = F.relu(self.dyn_conv(dynamics_batch))
#         # x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1).squeeze(-1)))
#         # x_6 = F.relu(self.buffer_fc(buff.squeeze(-1)))
#         qp = qp[:, :, -1]
#         skip = skip[:, :, -1]
#         re = re[:, :, -1]
#         buff = buff[:, :, -1]
#         x_5 = F.relu(self.knob_fc(torch.cat([re, skip, qp], dim=1)))
#         x_6 = F.relu(self.buffer_fc(buff))
#
#
#         x_1 = x_1.view(x_1.size(0), -1)
#         x_2 = x_2.view(x_2.size(0), -1)
#         x_3 = x_3.view(x_3.size(0), -1)
#         x_4 = x_4.view(x_4.size(0), -1)
#
#
#         x = torch.cat([x_5, x_6, x_1, x_2, x_3, x_4], dim=1)
#         x = F.relu(self.fc1(x))
#         critic = self.fc2(x)
#         return critic
#
#
# class NoiseLinear(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(NoiseLinear, self).__init__()
#         self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
#         self.sigma_weight = nn.Parameter(torch.ones(out_features, in_features) * 0.017)
#         self.mu_bias = nn.Parameter(torch.zeros(out_features))
#         self.sigma_bias = nn.Parameter(torch.ones(out_features) * 0.017)
#
#     def forward(self, x):
#         if self.training:
#             weight = self.mu_weight + self.sigma_weight * torch.randn_like(self.sigma_weight)
#             bias = self.mu_bias + self.sigma_bias * torch.randn_like(self.sigma_bias)
#         else:
#             weight = self.mu_weight
#             bias = self.mu_bias
#         return F.linear(x, weight, bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

S_INFO = 8  # bw, delay, buffer, qp, skip, r, segment_size, dynamics
S_LEN = 8  # past 8
A_DIM = 100
# A_DIM_1 = 5  # Output for (0-4)
# A_DIM_2 = 4  # Output for (0-3)
# A_DIM_3 = 4  # Output for (0-3)

# Define the Actor network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.input_channel = 1
        channel_cnn = 128
        channel_fc = 128
        # Define the actor's layers
        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.actor_conv3 = nn.Conv1d(3, channel_cnn, 1)  # L_out = 8 - (1-1) -1 + 1 = 8
        self.actor_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.actor_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        incoming_size = 4*channel_cnn*5 + channel_cnn*8 + 1
        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        self.fc2 = nn.Linear(in_features=channel_fc, out_features=A_DIM)

        # self.fc2_1 = nn.Linear(in_features=channel_fc, out_features=A_DIM_1)
        # self.fc2_2 = nn.Linear(in_features=channel_fc, out_features=A_DIM_2)
        # self.fc2_3 = nn.Linear(in_features=channel_fc, out_features=A_DIM_3)


    def forward(self, inputs):
        bw_batch = inputs[:, 0:1, :]
        delay_batch = inputs[:, 1:2, :]
        buff = inputs[:, 2:3, :]
        combined_batch = torch.cat([inputs[:, 3:4, :], inputs[:, 4:5, :], inputs[:, 5:6, :]], dim=1)
        size_batch = inputs[:, 6:7, :]
        dynamics_batch = inputs[:, 7:8, :]

        x_1 = F.relu(self.actor_conv1(bw_batch))
        x_2 = F.relu(self.actor_conv2(delay_batch))
        x_3 = F.relu(self.actor_conv3(combined_batch))
        x_4 = F.relu(self.actor_conv4(size_batch))
        x_5 = F.relu(self.actor_conv5(dynamics_batch))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))

        buff = buff[:, :, -1]
        buff = buff.view(buff.size(0), -1)

        x = torch.cat([x_1, x_2, x_3, x_4, x_5, buff], dim=1)
        x = F.relu(self.fc1(x))
        actor = F.softmax(self.fc2(x), dim=1)
        return actor
        # out1 = F.softmax(self.fc2_1(x), dim=1)
        # out2 = F.softmax(self.fc2_2(x), dim=1)
        # out3 = F.softmax(self.fc2_3(x), dim=1)
        # return out1, out2, out3

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_feature(self, inputs):
        bw_batch = inputs[:, 0:1, :]
        delay_batch = inputs[:, 1:2, :]
        buff = inputs[:, 2:3, :]
        combined_batch = torch.cat([inputs[:, 3:4, :], inputs[:, 4:5, :], inputs[:, 5:6, :]], dim=1)
        size_batch = inputs[:, 6:7, :]
        dynamics_batch = inputs[:, 7:8, :]

        x_1 = F.relu(self.actor_conv1(bw_batch))
        x_2 = F.relu(self.actor_conv2(delay_batch))
        x_3 = F.relu(self.actor_conv3(combined_batch))
        x_4 = F.relu(self.actor_conv4(size_batch))
        x_5 = F.relu(self.actor_conv5(dynamics_batch))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))

        buff = buff[:, :, -1]
        buff = buff.view(buff.size(0), -1)

        x = torch.cat([x_1, x_2, x_3, x_4, x_5, buff], dim=1)
        return x

# Define the Critic network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.input_channel = 1
        channel_cnn = 128
        channel_fc = 128
        # Define the actor's layers
        self.critic_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        self.critic_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.critic_conv3 = nn.Conv1d(3, channel_cnn, 1)  # L_out = 8 - (1-1) -1 + 1 = 8
        self.critic_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.critic_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        incoming_size = 4 * channel_cnn * 5 + channel_cnn * 8 + 1
        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        self.fc2 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        bw_batch = inputs[:, 0:1, :]
        delay_batch = inputs[:, 1:2, :]
        buff = inputs[:, 2:3, :]
        combined_batch = torch.cat([inputs[:, 3:4, :], inputs[:, 4:5, :], inputs[:, 5:6, :]], dim=1)
        size_batch = inputs[:, 6:7, :]
        dynamics_batch = inputs[:, 7:8, :]

        x_1 = F.relu(self.critic_conv1(bw_batch))
        x_2 = F.relu(self.critic_conv2(delay_batch))
        x_3 = F.relu(self.critic_conv3(combined_batch))
        x_4 = F.relu(self.critic_conv4(size_batch))
        x_5 = F.relu(self.critic_conv5(dynamics_batch))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))

        buff = buff[:, :, -1]
        buff = buff.view(buff.size(0), -1)

        x = torch.cat([x_1, x_2, x_3, x_4, x_5, buff], dim=1)
        x = F.relu(self.fc1(x))
        critic = self.fc2(x)
        return critic

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




