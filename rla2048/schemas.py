from pydantic import BaseModel, Field
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
    savefig_folder: Path


class TrajectoryStep(BaseModel):
    state: int
    action: int
    reward: float
    next_state: int


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
                    savefig_folder=Path('images'))
    return params
