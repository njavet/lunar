from selenium.webdriver.common.by import By
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import time
import numpy as np
from pathlib import Path

# project imports
from lunar.g2048.ffctrl import FirefoxControl
from lunar.g2048.agent import Agent
from lunar.g2048.dqn import ConNet
from lunar import Env2048
from lunar.agents.dqa import DQNAgent
from lunar.training import train_dqa
from lunar import config


def play_2048():
    fc = FirefoxControl()
    agent = Agent()
    try:
        fc.driver.get(fc.url)
        fc.game_container = fc.driver.find_element(By.TAG_NAME, 'body')
        fc.game_container.click()
        fc.game_message = fc.driver.find_element(By.CLASS_NAME, 'game-message')
        fc.score_element = fc.driver.find_element(By.CLASS_NAME, 'score-container')

        while True:
            board = fc.get_board()
            move = agent.find_best_move(board)
            fc.send_move(move)
            score = fc.get_score()
            print('SCORE', score)
            status = fc.get_status()
            if 'game over' in status.lower():
                break
            time.sleep(0.1)

        print('final board:')
        for row in board:
            print(row)
        print('final score:', score)
        time.sleep(5)
    finally:
        fc.driver.quit()


def get_params():
    epsilon_min = 0.01
    max_time_steps = 1000000
    decay = np.exp(np.log(epsilon_min) / max_time_steps)
    params = config.Params(n_envs=128,
                           gamma=1,
                           epsilon=1,
                           epsilon_min=epsilon_min,
                           max_time_steps=max_time_steps,
                           decay=decay,
                           batch_size=1024,
                           memory_size=5000000,
                           update_target_steps=1024,
                           checkpoint_steps=10000,
                           training_freq=1,
                           lr=5e-4,
                           seed=0x101,
                           eval_episodes=100,
                           model_file=Path('g2048.pth'),
                           video_folder=Path('videos'),
                           results_folder=Path('logs'))
    return params


def create_agent(params):
    agent = DQNAgent(gamma=params.gamma,
                     epsilon=params.epsilon,
                     epsilon_min=params.epsilon_min,
                     max_time_steps=params.max_time_steps,
                     decay=params.decay,
                     batch_size=params.batch_size,
                     memory_size=params.memory_size,
                     update_target_steps=params.update_target_steps,
                     checkpoint_steps=params.checkpoint_steps,
                     training_freq=params.training_freq,
                     lr=params.lr)
    return agent


def main():
    params = get_params()
    agent = create_agent(params)
    agent.init_dqn(ConNet)
    env = SubprocVecEnv([lambda: Env2048() for _ in range(params.n_envs)])
    env = VecMonitor(env)
    train_dqa(agent, env, params)
