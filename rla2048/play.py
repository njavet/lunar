import gymnasium as gym
import torch

# project imports
from rla2048.agents.dqn_agent import DQLAgent
from rla2048.schemas import get_default_params


def train():
    env = gym.make('rla2048/Game2048-v0', render_mode='human')
    params = get_default_params()
    agent = DQLAgent(env, params)
    agent.learn()
    torch.save(agent.model.state_dict(), 'dql_2048.pth')

