from pydantic import BaseModel
from pathlib import Path


class Params(BaseModel):
    n_envs: int = 256
    gamma: float = 1.0
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    max_time_steps: int = 1000000
    decay: float | None = None
    batch_size: int = 1024
    memory_size: int = 10000000
    update_target_steps: int = 1024
    training_freq: int = 1
    lr: float = 5e-4
    seed: int = 0x101
    eval_episodes: int = 10
    model_file: Path = Path('g2048.pth')
    video_folder: Path = Path('videos')
    results_folder: Path = Path('logs')


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
