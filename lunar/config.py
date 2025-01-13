from pydantic import BaseModel
from pathlib import Path


class Params(BaseModel):
    n_envs: int = 32
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    max_time_steps: int = 100000
    decay: float | None = None
    batch_size: int = 256
    memory_size: int = 2000000
    update_target_steps: int = 2048
    training_freq: int = 1
    lr: float = 1e-4
    seed: int = 0x101
    model_file: Path = Path('lunar.pth')
    video_folder: Path = Path('videos')
    model_folder: Path = Path('models')
