import torch
import time
import numpy as np
import gymnasium as gym



def load_checkpoint(agent, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.memory = checkpoint['memory']
    agent.epsilon = checkpoint['epsilon']


def save_checkpoint(agent, filename='checkpoint.pth'):
    dix = {'policy_state_dict': agent.policy_net.state_dict(),
           'target_state_dict': agent.target_net.state_dict(),
           'optimizer_state_dict': agent.optimizer.state_dict(),
           'memory': agent.memory.memory,
           'epsilon': agent.epsilon}
    torch.save(dix, filename)

