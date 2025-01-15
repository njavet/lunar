from pathlib import Path
import os
from pydantic import BaseModel
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

# project imports
from lunar.agent import Agent
from lunar.training import train_agent, evaluate_model
from lunar.vis import record_video, plot_evaluation




def create_video(model_file='models/lunar_small_g99.pth'):
    params = Params()
    agent = create_agent(params)
    agent.load_model(model_file)
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    record_video(agent, env, params.seed, params.video_folder)


def evaluation(model_file='models/lunar_small_g99.pth'):
    params = Params()
    agent = create_agent(params)
    agent.load_model(model_file)
    tr, ts, lr, rpa = evaluate_model(agent)
    plot_evaluation(tr, ts, lr, rpa)
    return tr, ts, lr, rpa


def main():
    params = Params()
    os.makedirs(params.video_folder, exist_ok=True)
    os.makedirs(params.results_folder, exist_ok=True)

    agent = create_agent(params)
    v_envs = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, v_envs, params)
