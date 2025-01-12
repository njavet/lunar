import torch
import time
from rich.console import Console
import gymnasium as gym

# project imports
from lunar.utils.tracker import Tracker
from lunar import config
from lunar.agents.a2084 import G2048Agent


def train_agent(agent, env, max_time_steps, n_envs, filename='lunar.pth'):
    tracker = Tracker(n_envs)
    console = Console()
    states = env.reset()
    start = time.time()
    for step in range(max_time_steps):
        actions = agent.select_actions(states)
        next_states, rewards, dones, infos = env.step(actions)
        tracker.update(agent.epsilon, rewards, dones)
        agent.store_transitions(states, actions, rewards, next_states, dones)
        agent.learn()
        states = next_states
        if step % 1000 == 0:
            agent.update_target_net()
        if step % 1000 == 0:
            int_time = time.time()
            tt = (int_time - start) / 60
            console.print(64*'-', style='blue')
            console.print(f'steps: {step}', style='cyan')
            console.print(f'current epsilon: {agent.epsilon:.4f}', style='#6312ff')
            console.print(f'Total Time: {tt:.2f} minutes...')
            logs = tracker.get_logs()
            console.print(f'mean reward: {logs['mean_reward']}')
            console.print(f'mean length: {logs['mean_length']}')
            console.print(f'episodes: {logs['episodes']}')
    torch.save(agent.target_net.state_dict(), filename)


def evaluate_policy(agent=None, fname='g2048.pth'):
    params = config.get_2048_params()
    agent = G2048Agent(gamma=params.gamma,
                       epsilon=params.epsilon,
                       epsilon_min=params.epsilon_min,
                       decay=params.decay,
                       batch_size=params.batch_size,
                       memory_size=params.memory_size,
                       update_target_steps=params.update_target_steps,
                       lr=params.lr)
    agent.target_net.load_state_dict(torch.load(fname))
    #env = gym.make('LunarLander-v3', render_mode='human')
    env = gym.make('Game2048-v0', render_mode='human')
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.optimal_policy(state)
        ns, r, term, trunc, info = env.step(action)
        print('reward', r)
        total_reward += r
        done = term or trunc
        state = ns
    print('TOT', total_reward)

