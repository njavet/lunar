from pydantic import BaseModel, Field
import torch


class OrchestratorParams(BaseModel):
    n_runs: int
    n_episodes: int
    seed: int


class LearnerParams(BaseModel):
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    decay: float
    batch_size: int
    memory_size: int
    update_target_steps: int


class TrajectoryStep(BaseModel):
    state: torch.Tensor
    action: int
    reward: torch.Tensor
    next_state: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class Trajectory(BaseModel):
    steps: list[TrajectoryStep] = Field(default_factory=list)
