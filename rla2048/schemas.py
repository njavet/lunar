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


def get_default_params():
    params = Params(n_runs=32,
                    n_episodes=2**14,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=0.8,
                    epsilon_min=0.05,
                    decay=0.99,
                    seed=0x101,
                    batch_size=64,
                    update_target_steps=10,
                    savefig_folder=Path('images'))
    return params
