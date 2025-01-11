import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch
import cv2

# project imports
from lunar.config import Params
from lunar.agent import DQNAgent
from lunar.dqns import DQN


def main():
    params = Params()
    agent = DQNAgent(dqn=DQN,
                     gamma=params.gamma,
                     epsilon=params.epsilon,
                     epsilon_min=params.epsilon_min,
                     decay=params.decay,
                     batch_size=params.batch_size,
                     memory_size=params.memory_size,
                     update_target_steps=params.update_target_steps,
                     lr=params.lr)
    env = make_vec_env('LunarLander-v3', n_envs=16)
    train_agent(agent, env)


def train_agent(agent, env):
    max_time_steps = 1000000
    states = env.reset()
    episode_rewards = np.zeros(16)
    for step in range(max_time_steps):
        actions = agent.select_actions(states)
        next_states, rewards, dones, infos = env.step(actions)
        episode_rewards += rewards
        agent.store_transitions(states, actions, rewards, next_states, dones)
        agent.learn()
        states = next_states
        if any(dones):
            current_rewards = [episode_rewards[i] for i in
                               range(len(episode_rewards)) if dones[i]]
            print(f'total rewards: ', np.mean(current_rewards))
            print(f'eps', agent.epsilon)
        if step % 1000 == 0:
            agent.update_target_net()
        if step % 100 == 0:
            print(f'step: {step}, rewards: {np.mean(episode_rewards)}')
    torch.save(agent.target_net.state_dict(), 'lunar.pth')


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
