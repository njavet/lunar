import torch
import time
import numpy as np
import gymnasium as gym
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_dim=8, action_dim=4):
        super(DQN, self).__init__()
        self.fc0 = nn.Linear(obs_dim, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc0(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_agent(agent, env, max_time_steps, n_envs):
    episode_rewards = np.zeros(n_envs)
    states = env.reset()
    start = time.time()
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
            int_time = time.time()
            tt = (int_time - start) / 60
            print(f'step: {step}, rewards: {np.mean(episode_rewards)}')
            print(f'eps', agent.epsilon)
            print(f'total time: {tt:.2f} min..')
    torch.save(agent.target_net.state_dict(), 'lunar_gpu_2.pth')


def evaluate_policy(fname='lunar.pth'):
    model = DQN()
    model.load_state_dict(torch.load(fname))
    env = gym.make('LunarLander-v3', render_mode='human')
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = model(state).argmax().cpu().numpy()
        #action = np.random.randint(0, 4)
        ns, r, term, trunc, info = env.step(action)
        print('reward', r)
        total_reward += r
        done = term or trunc
        state = ns
    print('TOT', total_reward)


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

