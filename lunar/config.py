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
    checkpoint_steps: int
    training_freq: int
    lr: float
    seed: int
    eval_episodes: int
    model_file: Path
    video_folder: Path
    results_folder: Path


TILE_COLORS = {
    0: '#ffffff',
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
