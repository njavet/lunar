from abc import ABC
import torch
from typing import Union
import gymnasium as gym

# project imports
from rla2048.schemas import OrchestratorParams, TrajectoryStep
from rla2048.agents.learner import Learner


class Orchestrator(ABC):
    def __init__(self,
                 env: gym.Env,
                 agent: Learner,
                 params: OrchestratorParams) -> None:
        self.env = env
        self.agent = agent
        self.params = params

    def exec_step(self, state: torch.Tensor) -> bool:
        action = self.agent.policy(state)
        next_state, reward, term, trunc, info = self.env.step(action)
        ts = TrajectoryStep(state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state)
        self.agent.trajectory.steps.append(ts)
        return term or trunc

    def process_step(self):
        pass

    def generate_trajectory(self, policy='optimal'):
        self.reset_trajectory()
        state, info = self.env.reset()
        terminated = False
        while not terminated:
            action = self.policies[policy](state)
            ts, terminated = self.exec_step(state, action)
            self.trajectory.steps.append(ts)
            self.process_step()
            state = ts.next_state

    def process_trajectory(self, episode):
        pass
