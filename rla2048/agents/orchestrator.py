from abc import ABC
import torch
import gymnasium as gym

# project imports
from rla2048.schemas import Trajectory, TrajectoryStep, Params


class Orchestrator(ABC):
    def __init__(self, env: gym.Env, params: Params):
        """ params could be seen as given by nature / god """
        self.env = env
        self.params = params
        self.trajectory = Trajectory()
        self.policies = {'optimal': self.optimal_policy,
                         'behave': self.behave_policy}

    def optimal_policy(self, state: torch.Tensor) -> int:
        raise NotImplementedError

    def behave_policy(self, state: torch.Tensor) -> int:
        raise NotImplementedError

    def reset_trajectory(self):
        self.trajectory = Trajectory()

    def exec_step(self,
                  state: torch.Tensor,
                  action: int) -> tuple[TrajectoryStep, bool]:
        next_state, reward, term, trunc, info = self.env.step(action)
        ts = TrajectoryStep(state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state)
        done = term or trunc
        return ts, done

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
