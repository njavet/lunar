import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict

# project imports
from rla2048.agents.learner import Learner
from rla2048.schemas import LearnerParams
from rla2048.dqn import DQN3


class DQLAgent(Learner):
    def __init__(self, params: LearnerParams):
        super().__init__(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.epsilon_decay = params.decay
        self.epsilon_min = params.epsilon_min
        self.batch_size = params.batch_size
        self.update_target_steps = params.update_target_steps
        self.memory = deque(maxlen=params.memory_size)
        # dqn
        self.model = DQN3(256, 4).to(self.device)
        self.target_model = DQN3(256, 4).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.trajectories = defaultdict(list)
        # collection infos

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        if np.random.rand() <= self.epsilon:
            return torch.randint(0, 4, (1,), device=self.device)
        with torch.no_grad():
            q_values = self.model(state)
        max_actions = (q_values == q_values.max()).nonzero(as_tuple=True)[0]
        return max_actions[torch.randint(0, len(max_actions), (1,), device='cuda')]

    def remember(self):
        ts = self.trajectory.steps[-1]
        self.memory.append(
            (ts.state, ts.action, ts.reward, ts.next_state, ts.done)
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: torch.stack(x).to(self.device),
            zip(*batch)
        )

        q_values = self.model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]

        targets = q_values.clone()
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * max_next_q_values[i]

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_step(self):
        self.remember()
        self.replay()
        if len(self.trajectory.steps) % self.update_target_steps == 0:
            self.update_target_model()

    def process_episode(self, episode):
        self.decay_epsilon()
        if episode % 50 == 0:
            print(f'episode {episode} with epsilon: {self.epsilon}')
            #so = sorted(self.max_tiles, reverse=True)[0]
            #print(f'Highest tile: {int(so[0])}, highest reward: {so[1]}')
            print(64*'-')
