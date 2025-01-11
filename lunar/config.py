from pydantic import BaseModel


class Params(BaseModel):
    obs_dim: int = 256
    action_dim: int = 4
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    decay: float = 0.999
    batch_size: int = 512
    memory_size: int = 5000000
    update_target_steps: int = 1024
    lr: float = 0.0005
