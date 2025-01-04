import torch.nn as nn
import torch.nn.functional as tf


class DQN3(nn.Module):
    def __init__(self, idim, odim):
        super(DQN3, self).__init__()
        self.fc0 = nn.Linear(idim, 512)
        self.fc1 = nn.Linear(512, idim)
        self.fc2 = nn.Linear(idim, odim)

    def forward(self, x):
        x0 = tf.relu(self.fc0(x))
        x1 = tf.relu(self.fc1(x0))
        x2 = tf.relu(self.fc2(x1))
        return x2


class DQN4(nn.Module):
    def __init__(self, idim, odim):
        super(DQN4, self).__init__()
        self.fc0 = nn.Linear(idim, 512)
        self.fc1 = nn.Linear(512, idim)
        self.fc2 = nn.Linear(idim, 128)
        self.fc3 = nn.Linear(128, odim)

    def forward(self, x):
        x0 = tf.relu(self.fc0(x))
        x1 = tf.relu(self.fc1(x0))
        x2 = tf.relu(self.fc2(x1))
        x3 = tf.relu(self.fc3(x2))
        return x3
