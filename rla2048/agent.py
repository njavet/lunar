from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 dqn: nn.Module,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 decay: float,
                 batch_size: int,
                 memory_size: int,
                 update_target_steps: int,
                 lr: float) -> None:
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_net = dqn(self.obs_dim, self.action_dim).to(self.dev)
        self.target_net = dqn(self.obs_dim, self.action_dim).to(self.dev)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(self.dev, memory_size=memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.steps = 0

    def select_actions(self, states):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim, size=len(states))
        states = torch.tensor(states, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            q_values = self.policy_net(states)
        actions = q_values.argmax(dim=1).detach().cpu().numpy()
        return actions

    def train(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)

    def store_transitions(self, states, actions, rewards, next_states, dones):
        self.memory.push(states, actions, rewards, next_states, dones)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class ReplayMemory:
    def __init__(self, device: torch.device, memory_size: int):
        self.device = device
        self.memory = deque(maxlen=memory_size)

    def push(self, states, actions, rewards, next_states, dones):
        transitions = zip(states, actions, rewards, next_states, dones)
        for transition in transitions:
            self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states),
                              dtype=torch.float32,
                              device=self.device)
        actions = torch.tensor(np.array(actions),
                               dtype=torch.long,
                               device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards),
                               dtype=torch.float32,
                               device=self.device)
        next_states = torch.tensor(np.array(next_states),
                                   dtype=torch.float32,
                                   device=self.device)
        dones = torch.tensor(np.array(dones),
                             dtype=torch.float32,
                             device=self.device)
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.memory)

"""
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

    def save_checkpoint(self, episode, filename='checkpoint.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,
            'episode': episode,
            'epsilon': self.epsilon
        }, filename)

"""