from pathlib import Path
from pydantic import BaseModel


class Params(BaseModel):
    n_envs: int
    gamma: float
    epsilon: float
    epsilon_min: float
    max_time_steps: int
    decay: float
    batch_size: int
    memory_size: int
    update_target_steps: int
    training_freq: int
    lr: float
    seed: int
    eval_episodes: int
    model_file: Path
    video_folder: Path
    results_folder: Path
