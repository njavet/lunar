import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch
import cv2

# project imports
from lunar import config
from lunar.agents.lunar_dql import (SmallLunarAgent,
                                    MiddleLunarAgent,
                                    LargeLunarAgent)


def train_small_agent():
    params = config.get_small_lunar_params()
    agent = SmallLunarAgent(gamma=params.gamma,
                            epsilon=params.epsilon,
                            epsilon_min=params.epsilon_min,
                            decay=params.decay,
                            batch_size=params.batch_size,
                            memory_size=params.memory_size,
                            update_target_steps=params.update_target_steps,
                            lr=params.lr)
    env = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, env, params.max_time_steps, params.n)


