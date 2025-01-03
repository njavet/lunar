import gymnasium as gym
import torch
from pathlib import Path
import cv2

# project imports
from rla2048.agents.dqn_agent import DQLAgent
from rla2048.schemas import Params


def get_default_params():
    params = Params(n_runs=32,
                    n_episodes=4024,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=1,
                    epsilon_min=0.05,
                    decay=0.99999999,
                    seed=0x101,
                    batch_size=64,
                    update_target_steps=10,
                    savefig_folder=Path('images'))
    return params


def play():
    env = gym.make('rla2048/Game2048-v0', render_mode='human')
    params = get_default_params()
    agent = DQLAgent(env, params)
    agent.model.load_state_dict(torch.load('dql_2048.pth', weights_only=False))
    agent.generate_trajectory()


def train():
    env = gym.make('rla2048/Game2048-v0', render_mode='rgb_array')
    params = get_default_params()
    agent = DQLAgent(env, params)
    agent.learn()
    torch.save(agent.model.state_dict(), 'dql_2048.pth')
    print(f'highest tile: {max(agent.max_tiles)}')

    agent.record_video()
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
