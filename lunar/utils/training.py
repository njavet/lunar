import torch
import numpy as np
import gymnasium as gym


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

