from pydantic import BaseModel


class Params(BaseModel):
    obs_dim: int = 8
    action_dim: int = 4
    n_envs: int = 32
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    decay: float = 0.9999
    batch_size: int = 512
    memory_size: int = 5000000
    update_target_steps: int = 1024
    lr: float = 0.0005
    max_time_steps: int = 10000


def get_small_lunar_params():
    params = Params()
    return params


def get_middle_lunar_params():
    params = Params()
    return params


def get_large_lunar_params():
    params = Params()
    return params
