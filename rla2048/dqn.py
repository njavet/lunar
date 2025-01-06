import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, idim, odim):
        super(DQN, self).__init__()
        self.fc0 = nn.Linear(idim, 512)
        self.fc1 = nn.Linear(512, idim)
        self.fc2 = nn.Linear(idim, 128)
        self.fc3 = nn.Linear(128, odim)

    def forward(self, x):
        x = nn.functional.relu(self.fc0(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
