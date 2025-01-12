import seaborn as sns
import matplotlib.pyplot as plt


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


def record_video():
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
