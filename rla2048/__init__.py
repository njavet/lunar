from gymnasium.envs.registration import register

register(
    id="rla2048/GridWorld-v0",
    entry_point="rla2048.envs:GridWorldEnv",
)
