import gc
from collections import defaultdict
import time
from rich.console import Console
import torch
import pandas as pd
import numpy as np


def train_agent(agent, env, params):
    tracker = Tracker(params.n_envs)
    states = env.reset()
    for step in range(params.max_time_steps):
        actions = agent.select_actions(states)
        next_states, rewards, dones, infos = env.step(actions)
        agent.store_transitions(states, actions, rewards, next_states, dones)
        loss = agent.learn()
        states = next_states
        tracker.update(agent.epsilon, rewards, dones, loss)
        if step % params.update_target_steps == 0:
            agent.update_target_net()
        if step % 1000 == 0:
            gc.collect()
            tracker.print_logs(step, agent.epsilon)
    torch.save(agent.target_net.state_dict(), params.model_file)
    tracker.save_logs()


def evaluate_policy(agent, env):
    total_reward = 0
    done = False
    state, _ = env.reset()
    while not done:
        action = agent.optimal_policy(state)
        next_state, reward, term, trunc, info = env.step(action)
        done = term or trunc
        state = next_state
        total_reward += reward


class Tracker:
    def __init__(self, n_envs):
        self.console = Console()
        self.start_t = time.time()
        self.n_envs = n_envs
        self.episode_rewards = np.zeros(n_envs)
        self.episode_lengths = np.zeros(n_envs)
        self.losses = []
        self.epsilons = []
        self.total_rewards = []
        self.total_lengths = []

    def update(self, epsilon, rewards, dones, loss):
        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i, done in enumerate(dones):
            if done:
                self.total_rewards.append(self.episode_rewards[i])
                self.total_lengths.append(self.episode_lengths[i])
                self.epsilons.append(epsilon)
                self.losses.append(loss)
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

    def save_logs(self):
        data = {'total_rewards': self.total_rewards,
                'mean_rewards': np.mean(self.total_rewards),
                'std_rewards': np.std(self.total_rewards),
                'total_lengths': self.total_lengths,
                'epsilons': self.epsilons}
        df = pd.DataFrame(data)
        df.to_csv('logs1.csv', index=False)

    def print_logs(self, step, epsilon):
        curr_time = time.time()
        diff_time = (curr_time - self.start_t) / 60
        self.console.print(64 * '-', style='blue')
        self.console.print(f'Total Time: {diff_time:.2f} minutes...')
        self.console.print(f'steps: {step} / episodes: {len(self.total_lengths)}')
        self.console.print(f'current epsilon: {epsilon:.4f}',
                           style='#6312ff')
        self.console.print(f'mean reward: {self.mean_reward}')
        self.console.print(f'std reward: {self.std_reward}')
        self.console.print(f'mean length: {self.mean_length}')

    @property
    def mean_reward(self):
        return int(np.mean(self.total_rewards) if self.total_rewards else 0)

    @property
    def mean_length(self):
        return int(np.mean(self.total_lengths) if self.total_lengths else 0)

    @property
    def std_reward(self):
        return int(np.std(self.total_rewards) if self.total_rewards else 0)
