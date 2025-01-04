from abc import ABC
from typing import Union
import torch
import numpy as np

# project imports
from rla2048.schemas import Params


class Learner(ABC):
    def __init__(self, params: Params, model=None):
        self.params = params
        self.model = model

    def policy(self, state: Union[torch.Tensor, np.ndarray, int]) -> int:
        raise NotImplementedError

    def process_step(self):
        raise NotImplementedError

    def process_trajectory(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
