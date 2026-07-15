import torch
import torch.nn as nn
import torch.nn.functional as F
from PPO import Actor
from IL import ILAgent

A_DIM = 100

class FusionActor(nn.Module):
    def __init__(self):
        super(FusionActor, self).__init__()
        self.fc_rl1 = nn.Linear(3585, 256)
        self.fc_rl2 = nn.Linear(256, 128)
        self.fc_il = nn.Linear(112, 128)
        # self.fc_merge = nn.Linear(256, 256)
        # self.fc_output = nn.Linear(256, A_DIM)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_features=A_DIM)

    def forward(self, frl, fil):
        frl = F.tanh(self.fc_rl1(frl))
        frl = F.tanh(self.fc_rl2(frl))
        fil = F.tanh(self.fc_il(fil))
        f_merge = torch.cat([frl, fil], dim=1)
        # f_softmax = F.softmax(self.fc_merge(f_merge), dim=1)
        # f_fusion = f_merge * f_softmax
        # action_distribution = F.softmax(self.fc_output(f_fusion), dim=1)
        # return action_distribution
        x = F.relu(self.fc1(f_merge))
        actor = F.softmax(self.fc2(x), dim=1)
        return actor


class FusionCritic(nn.Module):
    def __init__(self):
        super(FusionCritic, self).__init__()
        self.fc_rl1 = nn.Linear(3585, 256)
        self.fc_rl2 = nn.Linear(256, 128)
        self.fc_il = nn.Linear(112, 128)
        # self.fc_merge = nn.Linear(256, 256)
        # self.fc_output = nn.Linear(256, A_DIM)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_features=1)

    def forward(self, frl, fil):
        frl = F.tanh(self.fc_rl1(frl))
        frl = F.tanh(self.fc_rl2(frl))
        fil = F.tanh(self.fc_il(fil))
        f_merge = torch.cat([frl, fil], dim=1)
        # f_softmax = F.softmax(self.fc_merge(f_merge), dim=1)
        # f_fusion = f_merge * f_softmax
        # action_distribution = F.softmax(self.fc_output(f_fusion), dim=1)
        # return action_distribution
        x = F.relu(self.fc1(f_merge))
        critic = self.fc2(x)
        return critic