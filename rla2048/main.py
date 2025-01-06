import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch
import cv2

# project imports
from rla2048.config import Params
from rla2048.agent import DQNAgent
from rla2048.dqns import DQN


def main():
    params = Params()
    agent = DQNAgent(obs_dim=params.obs_dim,
                     action_dim=params.action_dim,
                     dqn=DQN,
                     gamma=params.gamma,
                     epsilon=params.epsilon,
                     epsilon_min=params.epsilon_min,
                     decay=params.decay,
                     batch_size=params.batch_size,
                     memory_size=params.memory_size,
                     update_target_steps=params.update_target_steps,
                     lr=params.lr)
    env = make_vec_env('Game2048-v0', n_envs=16)
    train_agent(agent, env)


def train_agent(agent, env):
    episode_rewards = np.zeros(env.num_envs)
    max_time_steps = 100000
    states = env.reset()
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
    torch.save(agent.target_net.state_dict(), 'g2048.pth')


def load_checkpoint(agent, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.memory = checkpoint['memory']
    agent.epsilon = checkpoint['epsilon']
    return checkpoint['episode']


def evaluate_policy(fname='dql_2048_cuda_2e10_heu.pth'):
    p = torch.load(fname)
    agent_params = get_learner_params()
    agent = DQLAgent(agent_params)
    env = gym.make('rla2048/Game2048-v0', render_mode='human')
    agent.model.load_state_dict(p)
    orch_params = get_orchestrator_params()
    orchestrator = Orchestrator(env, agent, orch_params)
    orchestrator.run_episode(0)



def record_video():
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
