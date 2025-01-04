import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict

# project imports
from rla2048.agents.learner import Learner
from rla2048.schemas import LearnerParams


class DQLAgent(Learner):
    def __init__(self, params: LearnerParams):
        super().__init__(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=500000)
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.epsilon_decay = params.decay
        self.epsilon_min = params.epsilon_min
        self.batch_size = params.batch_size
        self.update_target_steps = 16
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.trajectories = defaultdict(list)
        self.images = None
        self.max_tiles = None
        self.total_rewards = None

    def behave_policy(self, state: torch.Tensor) -> int:
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self,
                 state: torch.Tensor,
                 action: int,
                 reward: torch.Tensor,
                 next_state: torch.Tensor,
                 done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

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
        self.replay()
        if len(self.trajectory.steps) % self.update_target_steps == 0:
            self.update_target_model()

    def process_trajectory(self, episode):
        self.decay_epsilon()
        st = self.trajectory.steps[-1].next_state
        st = st.reshape((4, 4, 16)).cpu().numpy()
        inds = np.argwhere(st == 1)
        st = np.exp2(inds).astype(np.int32)
        tr = sum([ts.reward for ts in self.trajectory.steps])
        self.max_tiles.append((np.max(st), tr))
        self.total_rewards.append(tr)
        if episode % 100 == 0:
            print(f'episode {episode} with epsilon: {self.epsilon}')
            so = sorted(self.max_tiles, reverse=True)[0]
            print(f'Highest tile: {int(so[0])}, highest reward: {so[1]}')
            print(64*'-')

    def learn(self):
        self.max_tiles = []
        self.total_rewards = []
        for n in range(self.params.n_episodes):
            self.generate_trajectory(policy='behave')
            self.trajectories[n] = self.trajectory
            self.process_trajectory(n)
