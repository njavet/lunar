from pydantic import BaseModel


class Params(BaseModel):
    n_envs: int = 32
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    decay_proc: float = 0.8
    batch_size: int = 512
    memory_size: int = 1e6
    update_target_steps: int = 1024
    training_freq: int = 1
    lr: float = 0.001
    max_time_steps: int = 1e5
    filename: str = 'lunar.pth'
