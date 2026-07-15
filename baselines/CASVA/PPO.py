import torch
import torch.nn as nn
import torch.nn.functional as F

S_INFO = 8  # bw, latency, buffer, qp, skip, r, segment_size, dynamics
S_LEN = 8  # past 8
A_DIM = 100


# Define the Actor network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.input_channel = 1
        channel_cnn = 128
        channel_fc = 128
        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.actor_conv3 = nn.Conv1d(3, channel_cnn, 1)  # L_out = 8 - (1-1) -1 + 1 = 8
        self.actor_conv4 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        self.actor_conv5 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        incoming_size = 4*channel_cnn*5 + channel_cnn*8 + 1
        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        self.fc2 = nn.Linear(in_features=channel_fc, out_features=A_DIM)


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

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




