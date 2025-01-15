from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordVideo
import seaborn as sns
import matplotlib.pyplot as plt

# project imports
from lunar.agent import Agent


def plot_evaluation(rewards, steps, landings, rewards_per_action):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    plt.title('Mean Performance of the Trained Model in 10 Evaluation Episodes')
    mean_reward = np.mean(rewards)
    mean_steps = np.mean(steps)
    sns.barplot(x=list(range(1, 11)),
                y=rewards,
                palette='Blues',
                alpha=0.8,
                ax=ax[0, 0])
    ax[0, 0].axhline(y=mean_reward,
                     color='purple',
                     linestyle='--',
                     label=f'Mean Reward = {mean_reward:.2f}')
    ax[0, 0].set_title('Mean Reward')
    ax[0, 0].set_ylabel('Reward')
    sns.barplot(x=list(range(1, 11)),
                y=steps,
                palette='Blues',
                alpha=0.8,
                ax=ax[0, 1])
    ax[0, 1].axhline(y=mean_steps,
                     color='purple',
                     linestyle='--',
                     label=f'Mean Reward = {mean_steps:.2f}')
    ax[0, 1].set_title('Mean Steps')
    ax[0, 1].set_ylabel('Step')

    actions = ['Do nothing',
               'fire left orientation engine',
               'fire main engine',
               'fire right orientation engine']
    rpas = [rewards_per_action[0],
            rewards_per_action[1],
            rewards_per_action[2],
            rewards_per_action[3]]
    sns.barplot(x=actions, y=rpas, ax=ax[1, 0])
    ax[1, 0].set_xticklabels(actions, rotation=45)
    ax[1, 0].set_title('Reward per Action')
    ax[1, 0].set_ylabel('Reward')

    landing_labels = list(landings.keys())
    landing_counts = list(landings.values())
    sns.barplot(x=landing_labels, y=landing_counts, ax=ax[1, 1])
    ax[1, 1].set_title('Landing Results')
    ax[1, 1].set_ylabel('Number of Episodes')
    ax[1, 1].set_xticklabels(landing_labels, rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig('evaluation_plot.png')


def plot_training_results(logfile='logs/logs_small_g99.csv'):
    window_size = 100
    df = pd.read_csv(logfile)
    df['moving_avg_rewards'] = df['total_rewards'].rolling(window=window_size).mean()
    df['moving_avg_lengths'] = df['total_lengths'].rolling(window=window_size).mean()

    fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    sns.lineplot(data=df,
                 y='moving_avg_rewards',
                 x=range(len(df)),
                 ax=ax[0],
                 label='Mean Reward (Window = 100)')
    ax[0].set_title('Learning Curve: Mean Reward')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Mean Reward')

    sns.lineplot(data=df,
                 y='moving_avg_lengths',
                 x=range(len(df)),
                 ax=ax[1],
                 label='Episode Length (Window = 100)')
    ax[1].set_title('Episode Length')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Steps per Episode')

    sns.lineplot(data=df,
                 y='epsilons',
                 x=range(len(df)),
                 ax=ax[2],
                 label='Epsilon Decay')
    ax[2].set_title('Epsilon Decay')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig('training_plot.png')
    return


def plot_tracker(tracker):
    sns.set(style='darkgrid')
    # Plot rewards distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(tracker.total_rewards, kde=True, bins=20)
    plt.title('Distribution of Episode Rewards')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.savefig('rewards.png')

    # Plot episode lengths over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(tracker.total_lengths)), y=tracker.total_lengths)
    plt.title('Episode Lengths Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.savefig('episodes.png')


def record_video(agent: Agent, env: gym.Env, seed: int, video_path: Path):
    env = RecordVideo(env, video_folder=video_path.name)
    state, _ = env.reset(seed=seed)
    done = False
    tot = 0
    while not done:
        action = agent.optimal_policy(state)
        ns, reward, term, trunc, info = env.step(action)
        tot += reward
        state = ns
        done = term or trunc
    print('Total Reward', tot)
    env.close()
