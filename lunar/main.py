import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
import cv2

# project imports
from lunar import config
from lunar.training import train_agent, evaluate_policy
from lunar.utils.vis import record_video
from lunar.agents.lunar_dql import (SmallLunarAgent,
                                    MiddleLunarAgent,
                                    LargeLunarAgent)
from lunar.agents.a2084 import G2048Agent
from lunar.envs.g2048_env import Env2048


def create_small_agent():
    params = config.get_small_lunar_params()
    agent = SmallLunarAgent(gamma=params.gamma,
                            epsilon=params.epsilon,
                            epsilon_min=params.epsilon_min,
                            decay_proc=params.decay_proc,
                            batch_size=params.batch_size,
                            memory_size=params.memory_size,
                            update_target_steps=params.update_target_steps,
                            training_freq=params.training_freq,
                            max_time_steps=params.max_time_steps,
                            lr=params.lr)
    return agent, params


def train_agent(agent, params):
    env = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, env, params.max_time_steps, params.n_envs)
    return agent


def main():
    agent, params = create_small_agent()
    train_agent(agent, params)
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    record_video(agent, env, 1, '.')
    evaluate_policy(agent)


def train_large_agent():
    params = config.get_large_lunar_params()
    print('number of envs', params.n_envs)
    agent = LargeLunarAgent(gamma=params.gamma,
                            epsilon=params.epsilon,
                            epsilon_min=params.epsilon_min,
                            decay=params.decay,
                            batch_size=params.batch_size,
                            memory_size=params.memory_size,
                            update_target_steps=params.update_target_steps,
                            lr=params.lr)
    env = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, env, params.max_time_steps, params.n_envs)


def train_2048_agent():
    params = config.get_2048_params()
    print('number of envs', params.n_envs)
    agent = G2048Agent(gamma=params.gamma,
                       epsilon=params.epsilon,
                       epsilon_min=params.epsilon_min,
                       batch_size=params.batch_size,
                       memory_size=params.memory_size,
                       update_target_steps=params.update_target_steps,
                       max_time_steps=params.max_time_steps,
                       lr=params.lr)
    env = SubprocVecEnv([lambda: Env2048() for _ in range(params.n_envs)])
    train_agent(agent, env, params.max_time_steps, params.n_envs, 'g2048.pth')
