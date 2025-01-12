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
    memory_size: int = 1000000
    update_target_steps: int = 1024
    lr: float = 0.0005
    max_time_steps: int = 10000


def get_small_lunar_params():
    params = Params(n_envs=512,
                    decay=0.9995,
                    memory_size=1e6,
                    max_time_steps=1e3)
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


def get_2048_params():
    params = Params(obs_dim=256,
                    action_dim=4,
                    n_envs=128,
                    gamma=1,
                    batch_size=512,
                    memory_size=1e5,
                    max_time_steps=2e5
                    )
    return params


TILE_COLORS = {
    0: '#ffffff',     # Empty tile
    2: '#d5c7ff',
    4: '#c1adff',
    8: '#a384ff',
    16: '#9777f9',
    32: "#A259FF",
    64: '#8c61f6',
    128: '#8c61f6',
    256: '#8c61f6',
    512: '#b9a0f9',
    1024: '#b9a0f9',
    2048: '#b9a0f9',
    4096: '#6312ff'
}

TILE_SIZE = 101
GAP_SIZE = 8
WIDTH = 4 * TILE_SIZE + 5 * GAP_SIZE + 32
HEIGHT = WIDTH + 200
FONT_COLOR = 'black'

