import os
import gc
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

MAX_TIMESTEPS = 62500
PRINT_EVERY = 1250

LOG_DIR = "./runs"

N_TRAIN_ENV = 16
N_VAL_ENV = 16
N_TEST_ENV = 10

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(states):
            transition = (
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )
            self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self,
            state_size: int,
            action_size: int,
            device: str = "cpu",
            buffer_size: int = 100000,
            batch_size: int = 128,
            gamma: float = 0.99,
            lr: float = 0.001,
            ):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size

        # Networks
        self.qnetwork = QNetwork(state_size, action_size, 64).to(self.device)
        self.target_qnetwork = QNetwork(state_size, action_size, 64).to(self.device)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.target_qnetwork.eval()
        
        self.optimizer = Adam(
            params=self.qnetwork.parameters(),
            lr=lr,
        )
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)

        self.epsilon = 1.0
        self.epsilon_decay = 0.999926

    def act(self, states, use_epsilon=True):
        # If random exploration is chosen, pick random actions for the entire batch.
        if use_epsilon and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size, size=len(states))
        # Otherwise, choose actions greedily
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.qnetwork(states_tensor)
        actions = q_values.argmax(dim=1).detach().cpu().numpy()
        return actions

    def store_transition(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        self.optimizer.zero_grad()
        # Q-value for the chosen action
        q_values = self.qnetwork(states).gather(1, actions).squeeze()

        # Target Q-value
        with torch.no_grad():
            max_next_q_values = self.target_qnetwork(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

            
def main():
    run_path = os.path.join(LOG_DIR, "name")
    log_path = os.path.join(run_path, "logs")
    video_path = os.path.join(run_path, "videos")
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # Create the train environment
    train_env = make_vec_env('LunarLander-v2', n_envs=16)
    val_env = make_vec_env('LunarLander-v2', n_envs=10)
    
    video_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device="cuda"
    )

    total_rewards, epsilon_values, steps, val_rewards, val_std, val_steps = train_dqn(
        agent=agent,
        train_env=train_env,
        val_env=val_env,
        max_timesteps=MAX_TIMESTEPS,
        print_every=PRINT_EVERY
        )
    
    train_env.close()
    val_env.close()
    video_env.close()
    
    gc.collect()
    
    gen_plots(log_path, total_rewards, epsilon_values, steps, val_rewards, val_std, val_steps)
    
    eval_env = make_vec_env(
        'LunarLander-v2',
        n_envs=10
    )
    
    mean_test_reward, std_test_reward = evaluate_policy(
                                agent, 
                                env=eval_env
                            )
    print(f"Evaluation: {mean_test_reward:.2f} +/- {std_test_reward:.2f}")
    eval_env.close()
    gc.collect()

def gen_plots(save_dir, total_rewards, epsilon_values, steps, val_rewards, val_std, val_steps):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training data to CSV
    training_df = pd.DataFrame({
        'steps': steps,
        'epsilon': epsilon_values,
        'avg_reward': total_rewards
    })
    training_df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)

    # Save validation data to CSV
    validation_df = pd.DataFrame({
        'steps': val_steps,
        'mean_reward': val_rewards,
        'std_reward': val_std
    })
    validation_df.to_csv(os.path.join(save_dir, 'validation_results.csv'), index=False)

    # plot train history
    fig, ax1 = plt.subplots(figsize=(10,6))
    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Epsilon', color=color)
    ax1.plot(steps, epsilon_values, color=color, label='Epsilon')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Average Reward', color=color)  
    ax2.plot(steps, total_rewards, color=color, label='Average Training Reward')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Progress: Epsilon and Average Rewards Over Steps")
    fig.tight_layout()
    # Save the training plot
    train_plot_path = os.path.join(save_dir, "training_progress.png")
    fig.savefig(train_plot_path)
    plt.close(fig)

    # plot validation history
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Average Reward')

    val_rewards_arr = np.array(val_rewards)
    val_std_arr = np.array(val_std)

    # Plot the mean reward
    ax.plot(val_steps, val_rewards_arr, color='blue', label='Validation Reward (mean)')

    # Fill the area between mean and std with a lighter shade of blue
    ax.fill_between(val_steps, 
                    val_rewards_arr - val_std_arr, 
                    val_rewards_arr + val_std_arr, 
                    color='blue', alpha=0.2, label='Std Dev')

    ax.set_title("Validation Reward over Steps")
    ax.grid(True)
    ax.legend()

    # Save the validation plot
    val_plot_path = os.path.join(save_dir, "validation_progress.png")
    fig.savefig(val_plot_path)
    plt.close(fig)

def evaluate_policy(agent: DQNAgent, 
                    env: VecEnv,
                    seed: int,
                    use_epsilon=False):

    env.seed(seed)
    states = env.reset()    
    episode_rewards = np.zeros(env.num_envs)
    not_done = [True] * env.num_envs
    
    while True:
        actions = agent.act(states, use_epsilon=use_epsilon)
        next_states, rewards, dones, infos = env.step(actions)
        episode_rewards += (rewards * not_done)
        states = next_states
        
        for i in range(env.num_envs):
            if dones[i] and not_done[i]:
                not_done[i] = False
                
        if all(np.logical_not(not_done)):
            break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def train_dqn(
        agent: DQNAgent,
        train_env: VecEnv,
        val_env: VecEnv,
        video_env: VecEnv,
        video_path : str,
        max_timesteps: int,
        print_every: int,
    ):
    last_print_steps = 0
    total_rewards = []
    epsilon_values = []
    steps = []
    val_rewards = []
    val_std = []
    val_steps = []
    episode_rewards = np.zeros(train_env.num_envs)
    
    states = train_env.reset()

    for step in range(1, max_timesteps+1):
        actions = agent.act(states)
        next_states, rewards, dones, infos = train_env.step(actions)
        episode_rewards += rewards

        # replay buffer, had to handle the special case when the VecEnv automatically resets when done
        actual_next_states = []
        for i in range(train_env.num_envs):
            if dones[i]:
                actual_next_states.append(infos[i]["terminal_observation"])
            else:
                actual_next_states.append(next_states[i])
        agent.store_transition(states, actions, rewards, actual_next_states, dones)

        agent.learn()
        states = next_states

        # some logging and stuff
        if any(dones):
            current_rewards = [episode_rewards[i] for i in range(len(episode_rewards)) if dones[i]]
            total_rewards.append(np.mean(current_rewards))
            epsilon_values.append(agent.epsilon)
            steps.append(step)
            agent.update_target(len(current_rewards))

        episode_rewards = episode_rewards * np.logical_not(dones)
        
        if step % print_every == 0:
            avg_reward_since_last_print = np.mean(total_rewards[last_print_steps:])
            last_print_steps = len(total_rewards)
            mean_reward, std_reward = evaluate_policy(
                            agent=agent, 
                            env=val_env,
                    )
            val_rewards.append(mean_reward)
            val_std.append(std_reward)
            val_steps.append(step)
            
            current_e = agent.epsilon

    print("Training Complete!")
    return total_rewards, epsilon_values, steps, val_rewards, val_std, val_steps

def record_video(
        agent: DQNAgent,
        env: gym.Env,
        seed: int,
        video_path: str,
        step: int
    ):
    env = RecordVideo(env, video_folder=video_path, name_prefix=f"train_progress_{step}", disable_logger=True)
    obs, info = env.reset(seed=seed)
    o_shape: tuple = obs.shape
    o_shape = (1,) + o_shape
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.act(obs.reshape(o_shape), use_epsilon=False)[0]
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()

    
if __name__ == "__main__":
    main()