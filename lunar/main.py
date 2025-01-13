from pathlib import Path
import os
from pydantic import BaseModel
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

# project imports
from lunar.agent import Agent
from lunar.training import train_agent, evaluate_model
from lunar.vis import record_video


class Params(BaseModel):
    n_envs: int = 32
    gamma: float = 0.98
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    max_time_steps: int = 500000
    decay: float | None = None
    batch_size: int = 512
    memory_size: int = 5000000
    update_target_steps: int = 1024
    training_freq: int = 4
    lr: float = 5e-4
    seed: int = 0x101
    eval_episodes: int = 10
    model_file: Path = Path('lunar_large_0.pth')
    video_folder: Path = Path('videos')
    results_folder: Path = Path('results')


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


def main():
    params = Params()
    os.makedirs(params.video_folder, exist_ok=True)
    os.makedirs(params.results_folder, exist_ok=True)

    agent = create_agent(params)
    v_envs = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, v_envs, params)
    #eval_env = gym.make('LunarLander-v3', render_mode='human')
    #evaluate_policy(agent, eval_env)
    #agent.load_model('lunar.pth')

    video_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    record_video(agent, video_env, params.seed, params.video_folder)
