import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict

# project imports
from rla2048.agents.schopenhauer import SchopenhauerAgent


class DQLAgent(SchopenhauerAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=2000)
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

    def optimal_policy(self, state: np.ndarray) -> int:
        if np.random.rand() <= 0.1:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        print('Q values', q_values)
        return torch.argmax(q_values).item()

    def behave_policy(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    @staticmethod
    def create_model():
        model = nn.Sequential(nn.Linear(256, 512),
                              nn.ReLU(),
                              nn.Linear(512, 256),
                              nn.ReLU(),
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128, 4))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            target = self.model(state_tensor.unsqueeze(0)).detach().clone()
            if done:
                target[0][action] = reward
            else:
                next_q_values = self.target_model(next_state_tensor.unsqueeze(0)).detach()
                target[0][action] = reward + self.gamma * torch.max(next_q_values).item()
            states.append(state_tensor)
            targets.append(target[0])

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_step(self, record=False):
        if record:
            img = self.env.render()
            self.images.append(img)

        self.replay()
        if len(self.trajectory.steps) % self.update_target_steps == 0:
            self.update_target_model()

    def process_trajectory(self, episode):
        self.decay_epsilon()
        st = self.trajectory.steps[-1].next_state
        st = st.reshape((4, 4, 16))
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

    def record_video(self):
        self.images = []
        self.generate_trajectory(policy='behave', record=True)
