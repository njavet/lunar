import gymnasium as gym
import os
from stable_baselines3.common.env_util import make_vec_env

# project imports
from lunar.config import Params
from lunar.agent import Agent
from lunar.training import train_agent, evaluate_policy
from lunar.vis import record_video


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
    os.makedirs(params.model_folder, exist_ok=True)

    agent = create_agent(params)
    v_envs = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    #train_agent(agent, v_envs, params)
    #eval_env = gym.make('LunarLander-v3', render_mode='human')
    #evaluate_policy(agent, eval_env)
    agent.load_model('lunar.pth')

    video_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    record_video(agent, video_env, params.seed, params.video_folder)
