import gymnasium as gym


def get_env(params):
    env = gym.make('Game2048-v0', render_mode='human')
    params.state_size = env.observation_space.n
    params.action_size = env.action_space.n
    return env
