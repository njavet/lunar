from .env import Env2048
from gymnasium.envs.registration import register

register(
    id='rla2048/Game2048-v0',
    entry_point='rla2048:Env2048',
)

