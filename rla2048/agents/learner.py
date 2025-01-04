from abc import ABC
from typing import Union
import torch
import numpy as np

# project imports
from rla2048.schemas import LearnerParams, Trajectory, TrajectoryStep


class Learner(ABC):
    def __init__(self, params: LearnerParams, model=None):
        self.params = params
        self.model = model
        self.trajectory: Trajectory = Trajectory()

    def reset_trajectory(self):
        self.trajectory = Trajectory()

    def policy(self, state: Union[torch.Tensor, np.ndarray, int]) -> int:
        raise NotImplementedError

    def process_step(self):
        raise NotImplementedError

    def process_trajectory(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
