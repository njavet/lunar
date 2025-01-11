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


def train_agent(agent, env, max_time_steps, n_envs):
    episode_rewards = np.zeros(n_envs)
    states = env.reset()
    for step in range(max_time_steps):
        actions = agent.select_actions(states)
        next_states, rewards, dones, infos = env.step(actions)
        episode_rewards += rewards
        agent.store_transitions(states, actions, rewards, next_states, dones)
        agent.learn()
        states = next_states
        if step % 1000 == 0:
            agent.update_target_net()
        if step % 1000 == 0:
            print(f'step: {step}, rewards: {np.mean(episode_rewards)}')
            print(f'eps', agent.epsilon)
    torch.save(agent.target_net.state_dict(), 'lunar.pth')



def evaluate_policy(fname='lunar.pth'):
    p = torch.load(fname)
    model = p['target_state_dict']
    env = gym.make('LunarLander-v3', render_mode='human')
    done = False
    state, _ = env.reset()
    while not done:
        action = model(state).argmax()
        ns, r, term, trunc, info = env.step(action)
        done = term or trunc
        state = ns


def record_video():
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
