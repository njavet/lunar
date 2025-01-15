import torch.nn as nn

# project imports
from lunar import config
from lunar.agents.dqa import DQNAgent


class LunarDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


class LargeLunarDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


def get_params():
    params = config.Params()
    return params


def create_agent(params):
    agent = DQNAgent(gamma=params.gamma,
                     epsilon=params.epsilon,
                     epsilon_min=params.epsilon_min,
                     max_time_steps=params.max_time_steps,
                     decay=params.decay,
                     batch_size=params.batch_size,
                     memory_size=params.memory_size,
                     update_target_steps=params.update_target_steps,
                     training_freq=params.training_freq,
                     lr=params.lr)
    return agent


def lunar():
    params = get_params()
    agent = create_agent(params)

