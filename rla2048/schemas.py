from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path


class Params(BaseModel):
    n_runs: int
    n_episodes: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    decay: float
    seed: int
    batch_size: int
    update_target_steps: int
    savefig_folder: Path


class TrajectoryStep(BaseModel):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class Trajectory(BaseModel):
    steps: list[TrajectoryStep] = Field(default_factory=list)


