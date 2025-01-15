from collections import defaultdict
import gc
import time
from rich.console import Console
import torch
import pandas as pd
import numpy as np


def train_dqa(agent, env, params):
    tracker = Tracker(params.n_envs, window_size=1000)
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
            tracker.print_logs(step, agent.epsilon, infos)
    torch.save(agent.target_net.state_dict(), params.model_file)
    tracker.save_logs()


def evaluate_model(agent, env, n_episodes=10):
    total_rewards = []
    total_steps = []
    rewards_per_action = defaultdict(float)
    action_count = defaultdict(int)
    landing_results = {'crashed': 0,
                       'failed': 0,
                       'landed': 0,
                       'perfect': 0}

    for episode in range(n_episodes):
        episode_reward = 0
        steps = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.optimal_policy(state)
            next_state, reward, term, trunc, infos = env.step(action)
            episode_reward += reward
            steps += 1
            done = term or trunc
            state = next_state

            # collecting infos
            rewards_per_action[action.item()] += reward
            action_count[action.item()] += 1

        # collecting infos
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        if episode_reward < 0:
            landing_results['crashed'] += 1
        elif episode_reward < 200:
            landing_results['failed'] += 1
        elif 200 <= episode_reward < 299:
            landing_results['landed'] += 1
        else:
            landing_results['perfect'] += 1

    for action, count in action_count.items():
        rewards_per_action[action] /= count

    return total_rewards, total_steps, landing_results, rewards_per_action


class Tracker:
    def __init__(self, n_envs, window_size):
        self.console = Console()
        self.start_t = time.time()
        self.window_size = window_size
        self.n_envs = n_envs
        self.episode_rewards = np.zeros(n_envs)
        self.episode_lengths = np.zeros(n_envs)
        self.tiles = defaultdict(int)

        self.epsilons = []
        self.total_rewards = []
        self.total_lengths = []
        self.mean_rewards = []
        self.mean_lengths = []

    def update(self, epsilon, rewards, dones, infos):
        self.episode_rewards += rewards
        self.episode_lengths += 1
        for info in infos:
            for tile, count in info['tiles'].items():
                self.tiles[tile] += count
        self.tiles = {k: v / self.n_envs for k, v in self.tiles.items()}

        for i, done in enumerate(dones):
            if done:
                self.total_rewards.append(self.episode_rewards[i])
                self.total_lengths.append(self.episode_lengths[i])
                self.epsilons.append(epsilon)
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                self.mean_rewards.append(np.mean(self.total_rewards))
                self.mean_lengths.append(np.mean(self.total_lengths))

    def save_logs(self):
        data = {'total_rewards': self.total_rewards,
                'mean_rewards': self.mean_rewards,
                'total_lengths': self.total_lengths,
                'mean_lengths': self.mean_lengths,
                'epsilons': self.epsilons}
        df = pd.DataFrame(data)
        df.to_csv('logs.csv', index=False)

    def print_logs(self, step, epsilon, infos):
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
        self.console.print(f'tiles: {self.tiles}')

    @property
    def mean_reward(self):
        return np.mean(self.total_rewards[-self.window_size:])

    @property
    def mean_length(self):
        return np.mean(self.total_lengths[-self.window_size:])

    @property
    def std_reward(self):
        return np.std(self.total_rewards[-self.window_size:])
