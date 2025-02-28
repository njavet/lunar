from collections import deque
import random
import numpy as np
import torch
import torch.optim as optim


class DQNAgent:
    def __init__(self,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 max_time_steps: int,
                 decay: float,
                 batch_size: int,
                 memory_size: int,
                 update_target_steps: int,
                 checkpoint_steps: int,
                 training_freq: int,
                 lr: float) -> None:
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = ReplayMemory(self.dev, memory_size=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.max_time_steps = max_time_steps
        self.decay = decay
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps
        self.checkpoint_steps = checkpoint_steps
        self.training_freq = training_freq
        self.lr = lr
        self.steps = 0
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.reward_normalizer = RewardNormalizer()

    def init_dqn(self, dqn):
        self.policy_net = dqn().to(self.dev)
        self.target_net = dqn().to(self.dev)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.load_checkpoint()

    def load_model(self, filename):
        self.target_net.load_state_dict(torch.load(filename))

    def save_state(self):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'memory': self.memory.memory,
        }, 'checkpoint.pth')

    def load_checkpoint(self):
        try:
            checkpoint = torch.load('checkpoint.pth', map_location=self.dev)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.steps = checkpoint['steps'] + 1
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = checkpoint['epsilon']
            self.memory.memory = checkpoint.get('memory', deque(maxlen=self.memory.memory.maxlen))
            print('checkpoint loaded...')
        except FileNotFoundError:
            print('no checkpoint found ...')

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

    def optimal_policy(self, states):
        states = torch.tensor(states, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            q_values = self.target_net(states)
        try:
            actions = q_values.argmax(dim=1)
        except IndexError:
            actions = q_values.argmax()
        return actions.cpu().numpy()

    def select_actions(self, states):
        if random.random() < self.epsilon:
            action = np.random.randint(0, 4, len(states))
            return action
        states = torch.tensor(states, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            q_values = self.policy_net(states)
        actions = q_values.argmax(dim=1)
        return actions.cpu().numpy()

    def learn(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return
        if self.steps % self.training_freq != 0:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values *= (1 - dones)
        expected_q_values = rewards + self.gamma * next_q_values
        loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon_decay()

    def normalize_rewards(self, rewards):
        normalized_rewards = [self.reward_normalizer.normalize(r) for r in rewards]
        for r in rewards:
            self.reward_normalizer.update(r)
        return normalized_rewards

    def store_transitions(self, states, actions, rewards, next_states, dones):
        normalized_rewards = self.normalize_rewards(rewards)
        self.memory.push(states, actions, rewards, next_states, dones)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class ReplayMemory:
    def __init__(self, device: torch.device, memory_size: int):
        self.device = device
        self.memory = deque(maxlen=memory_size)

    def push(self, states, actions, rewards, next_states, dones):
        transitions = zip(states, actions, rewards, next_states, dones)
        self.memory.extend(list(transitions))

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


class RewardNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.count = 1e-4

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward):
        std = (self.var / self.count) ** 0.5
        return (reward - self.mean) / (std + 1e-8)
