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
    params = Params(n_envs=256,
                    decay=0.99999,
                    max_time_steps=1e5)
    return params


def get_middle_lunar_params():
    params = Params(n_envs=256)
    return params


def get_large_lunar_params():
    params = Params(n_envs=2048,
                    decay=0.9999,
                    batch_size=1024,
                    memory_size=1e6,
                    max_time_steps=50000)
    return params
