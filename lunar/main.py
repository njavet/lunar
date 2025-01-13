import gymnasium as gym
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
                  decay_proc=params.decay_proc,
                  batch_size=params.batch_size,
                  memory_size=params.memory_size,
                  update_target_steps=params.update_target_steps,
                  training_freq=params.training_freq,
                  max_time_steps=params.max_time_steps,
                  lr=params.lr)
    return agent


def main():
    params = Params()
    agent = create_agent(params)
    v_envs = make_vec_env('LunarLander-v3', n_envs=params.n_envs)
    train_agent(agent, v_envs, params)
    eval_env = gym.make('LunarLander-v3', render_mode='human')
    evaluate_policy(agent, eval_env)

    video_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    record_video(agent, video_env, 1, '.')
