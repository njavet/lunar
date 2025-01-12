import torch
import torch.optim as optim

# project imports
from lunar.agents.base_dql import DQNAgent
from lunar.dqns import ConNet


class G2048Agent(DQNAgent):
    def __init__(self,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 batch_size: int,
                 memory_size: int,
                 update_target_steps: int,
                 max_time_steps: int,
                 lr: float) -> None:
        super().__init__(gamma,
                         epsilon,
                         epsilon_min,
                         batch_size,
                         memory_size,
                         update_target_steps,
                         max_time_steps,
                         lr)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_dqn(self):
        self.policy_net = ConNet().to(self.dev)
        self.target_net = ConNet().to(self.dev)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
