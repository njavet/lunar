from abc import ABC
import torch
import gymnasium as gym

# project imports
from rla2048.schemas import Trajectory, TrajectoryStep, Params


class SchopenhauerAgent(ABC):
    """
    For now lets define a SchopenhauerAgent as an Agent
    that has an environment as part of himself. So the environment exists
    only inside the agent. Another type would be a Cartesian Agent that is
    part of the environment. The third Agent type would be a mix of both.
    """
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

    def process_step(self, record=False):
        pass

    def generate_trajectory(self, policy='optimal', record=False):
        self.reset_trajectory()
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).to('cuda')
        terminated = False
        while not terminated:
            action = self.policies[policy](state)
            ts, terminated = self.exec_step(state, action)
            self.trajectory.steps.append(ts)
            # the agent might want to do something after each step
            self.process_step(record)
            state = ts.next_state

    def process_trajectory(self, episode):
        pass
