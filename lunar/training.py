import torch
import time
from rich.console import Console
import gymnasium as gym

# project imports
from lunar.utils.tracker import Tracker


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
            console.print(f'steps: {step}', style='cyan')
            console.print(f'current epsilon: {agent.epsilon:.4f}', style='#6312ff')
            console.print(f'Total Time: {tt:.2f} minutes...')
            logs = tracker.get_logs()
            console.print(f'mean reward: {logs['mean_reward']}')
            console.print(f'mean length: {logs['mean_length']}')
            console.print(f'episodes: {logs['episodes']}')
    torch.save(agent.target_net.state_dict(), filename)


def evaluate_policy(agent, fname='lunar_01.pth'):
    agent.target_net.load_state_dict(torch.load(fname))
    env = gym.make('LunarLander-v3', render_mode='human')
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

