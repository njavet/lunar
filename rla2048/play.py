import gymnasium as gym
import torch
from pathlib import Path

# project imports
from rla2048.agents.dqn_agent import DQLAgent
from rla2048.schemas import Params


def get_default_params():
    params = Params(n_runs=32,
                    n_episodes=1,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=0.8,
                    epsilon_min=0.05,
                    decay=0.99,
                    seed=0x101,
                    batch_size=64,
                    update_target_steps=10,
                    savefig_folder=Path('images'))
    return params


def train():
    env = gym.make('rla2048/Game2048-v0', render_mode='human')
    params = get_default_params()
    agent = DQLAgent(env, params)
    agent.learn()
    torch.save(agent.model.state_dict(), 'dql_2048.pth')

