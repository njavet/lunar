import gymnasium as gym
import datetime
import torch
from pathlib import Path
import cv2

# project imports
from rla2048.agents.dqn_agent import DQLAgent
from rla2048.schemas import Params


def get_default_params():
    params = Params(n_runs=32,
                    n_episodes=2**20,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=1,
                    epsilon_min=0.05,
                    decay=0.999999,
                    seed=0x101,
                    batch_size=1024,
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
    start = datetime.datetime.now()
    agent.learn()
    end = datetime.datetime.now()
    print(f'highest tile: {max(agent.max_tiles)}')
    print('training time: ', (end - start))
    torch.save(agent.model.state_dict(), 'dql_2048_cuda_2e20.pth')

    return
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
