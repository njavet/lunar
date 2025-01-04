import gymnasium as gym
import torch
import cv2

# project imports
from rla2048.agents.dqn_agent import DQLAgent
from rla2048.agents.orchestrator import Orchestrator
from rla2048.schemas import OrchestratorParams, LearnerParams


def get_learner_params():
    params = LearnerParams(alpha=0.1,
                           gamma=0.99,
                           epsilon=1.0,
                           epsilon_min=0.05,
                           decay=0.998,
                           batch_size=512,
                           memory_size=1000000,
                           update_target_steps=101)
    return params


def get_orchestrator_params():
    params = OrchestratorParams(n_runs=0,
                                n_episodes=2**11,
                                seed=0x101)
    return params


def train():
    env = gym.make('rla2048/Game2048-v0', render_mode='rgb_array')
    agent_params = get_learner_params()
    agent = DQLAgent(agent_params)
    orch_params = get_orchestrator_params()
    orchestrator = Orchestrator(env, agent, orch_params)
    orchestrator.train_agent()
    torch.save(agent.target_model.state_dict(), 'dql_2048_cuda_2e11.pth')


def record_video():
    height, width = agent.images[0].shape[0], agent.images[0].shape[1]
    video = cv2.VideoWriter('dqn2048.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            4,
                            (width, height))
    for img in agent.images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
