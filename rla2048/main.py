import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# project imports
from rla2048 import config
from rla2048.agent import Agent
from rla2048.training import train_agent, evaluate_model
from rla2048.env import Env2048
from rla2048.vis import record_video, plot_evaluation


def create_agent(params):
    agent = Agent(gamma=params.gamma,
                  epsilon=params.epsilon,
                  epsilon_min=params.epsilon_min,
                  max_time_steps=params.max_time_steps,
                  decay=params.decay,
                  batch_size=params.batch_size,
                  memory_size=params.memory_size,
                  update_target_steps=params.update_target_steps,
                  training_freq=params.training_freq,
                  lr=params.lr)
    return agent


def create_video(model_file='models/g2048.pth'):
    params = config.Params()
    agent = create_agent(params)
    agent.load_model(model_file)
    env = gym.make('Game2048-v0', render_mode='rgb_array')
    record_video(agent, env, params.seed, params.video_folder)


def evaluation(model_file='models/g2048.pth'):
    params = config.Params()
    agent = create_agent(params)
    agent.load_model(model_file)
    env = gym.make('Game2048-v0', render_mode='rgb_array')
    tr, ts, lr, rpa = evaluate_model(agent, env)
    plot_evaluation(tr, ts, lr, rpa)
    return tr, ts, lr, rpa


def main():
    params = config.Params()
    os.makedirs(params.video_folder, exist_ok=True)
    os.makedirs(params.results_folder, exist_ok=True)

    agent = create_agent(params)
    env = SubprocVecEnv([lambda: Env2048() for _ in range(params.n_envs)])
    train_agent(agent, env, params)
