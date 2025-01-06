import gymnasium as gym
import torch
import cv2

# project imports
from rla2048.agent import DQNAgent



def train():
    env = gym.make('rla2048/Game2048-v0', render_mode='rgb_array')
    agent_params = get_learner_params()
    agent = DQLAgent(agent_params)
    orch_params = get_orchestrator_params()
    orchestrator = Orchestrator(env, agent, orch_params)
    orchestrator.train_agent()
    torch.save(agent.target_model.state_dict(), 'dql_2048_cuda_2e10_heu.pth')


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
