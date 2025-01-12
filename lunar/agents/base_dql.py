from abc import ABC
from collections import deque
import random
import numpy as np
import torch


class DQNAgent(ABC):
    def __init__(self,
                 gamma: float,
                 epsilon: float,
                 epsilon_min: float,
                 decay_proc: float,
                 batch_size: int,
                 memory_size: int,
                 update_target_steps: int,
                 max_time_steps: int,
                 lr: float) -> None:
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = ReplayMemory(self.dev, memory_size=memory_size)
        # self.memory = ReplayMemoryGPU(self.dev, memory_size=memory_size, state_shape=(8,))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_proc = decay_proc
        self.max_time_steps = max_time_steps
        self.decay_steps = self.compute_decay_steps()
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps
        self.lr = lr
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.init_dqn()
        self.steps = 0

    def init_dqn(self):
        raise NotImplementedError

    def compute_decay_steps(self):
        tol = 0.0001
        a = self.decay_proc * self.max_time_steps
        b = np.log(tol) * -self.decay_proc
        return int(a / b)

    def epsilon_decay(self):
        tmp = (1 - self.epsilon_min) * np.exp(-self.steps / self.decay_steps)
        self.epsilon = self.epsilon_min + tmp

    def optimal_policy(self, states):
        states = torch.tensor(states, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            q_values = self.target_net(states)
        actions = q_values.argmax(dim=1)
        return actions.cpu().numpy()

    def select_actions(self, states):
        if random.random() < self.epsilon:
            return torch.randint(0, 4, (len(states),), device=self.dev).cpu().numpy()
        states = torch.tensor(states, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            q_values = self.policy_net(states)
        actions = q_values.argmax(dim=1)
        return actions.cpu().numpy()

    def learn(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        # GPU
        #q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon_decay()

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


class ReplayMemoryGPU:
    def __init__(self, device: torch.device, memory_size: int, state_shape: tuple):
        self.device = device
        self.capacity = memory_size
        self.index = 0
        self.full = False

        # Pre-allocate memory on GPU
        self.states = torch.zeros((memory_size, *state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(memory_size, 1, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(memory_size, 1, dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((memory_size, *state_shape), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(memory_size, 1, dtype=torch.float32, device=self.device)

    def push(self, states, actions, rewards, next_states, dones):
        size = len(states)
        if self.capacity < self.index + size:
            self.full = True
            self.index = 0
            end = size
        else:
            end = self.index + size

        self.states[self.index:end] = torch.tensor(states, device=self.device)
        self.actions[self.index:end] = torch.tensor(actions, device=self.device).unsqueeze(1)
        self.rewards[self.index:end] = torch.tensor(rewards, device=self.device).unsqueeze(1)
        self.next_states[self.index:end] = torch.tensor(next_states, device=self.device)
        self.dones[self.index:end] = torch.tensor(dones, device=self.device).unsqueeze(1)

        self.index = end

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.index
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices].squeeze(1),
            self.rewards[indices].squeeze(1),
            self.next_states[indices],
            self.dones[indices].squeeze(1)
        )

    def __len__(self):
        return self.capacity if self.full else self.index
