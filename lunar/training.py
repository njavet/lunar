import gc
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
        agent.learn()
        states = next_states
        tracker.update(agent.epsilon, rewards, dones, infos)
        if step % params.update_target_steps == 0:
            agent.update_target_net()
        if step % 1000 == 0:
            gc.collect()
            tracker.print_logs(step)
    torch.save(agent.target_net.state_dict(), params.filename)
    tracker.save_logs()


def evaluate_policy(agent, env, filename=None):
    if filename is not None:
        agent.target_net.load_state_dict(torch.load(filename))
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
    def __init__(self, num_envs):
        self.console = Console()
        self.start_t = time.time()
        self.num_envs = num_envs
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)
        self.epsilons = []
        self.total_rewards = []
        self.total_lengths = []

    def update(self, epsilon, rewards, dones, infos):
        self.epsilons.append(epsilon)
        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i, done in enumerate(dones):
            if done:
                self.total_rewards.append(self.episode_rewards[i])
                self.total_lengths.append(self.episode_lengths[i])
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

    def save_logs(self):
        df0 = pd.DataFrame({'total_rewards': self.total_rewards})
        df1 = pd.DataFrame({'total_lengths': self.total_lengths})
        df2 = pd.DataFrame({'epsilons': self.epsilons})
        df0.to_csv('total_rewards.csv')
        df1.to_csv('total_length.csv')
        df2.to_csv('epsilons.csv')

    def print_logs(self, step):
        curr_time = time.time()
        diff_time = (curr_time - self.start_t) / 60
        self.console.print(64 * '-', style='blue')
        self.console.print(f'Total Time: {diff_time:.2f} minutes...')
        self.console.print(f'steps: {step} / episodes: {len(self.total_lengths)}')
        self.console.print(f'current epsilon: {self.epsilons[-1]:.4f}',
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
