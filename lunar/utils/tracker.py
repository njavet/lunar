import numpy as np


class Tracker:
    def __init__(self, num_envs):
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

    def get_logs(self):
        mean_reward = np.mean(self.total_rewards) if self.total_rewards else 0
        mean_length = np.mean(self.total_lengths) if self.total_lengths else 0
        episodes = len(self.total_lengths)
        return {'mean_reward': mean_reward,
                'mean_length': mean_length,
                'episodes': episodes}
