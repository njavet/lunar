import seaborn as sns
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def plot_tracker(tracker):
    sns.set(style='darkgrid')
    # Plot rewards distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(tracker.total_rewards, kde=True, bins=20)
    plt.title("Distribution of Episode Rewards")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.savefig('rewards.png')

    # Plot episode lengths over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(tracker.total_lengths)), y=tracker.total_lengths)
    plt.title("Episode Lengths Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.savefig('episodes.png')


def record_video(agent, env: gym.Env, seed: int, video_path: str):
    env = RecordVideo(env, video_folder=video_path)
    state, _ = env.reset(seed=seed)
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.optimal_policy(state)
        ns, reward, term, trunc, info = env.step(action)
        state = ns
        done = term or trunc

    env.close()
