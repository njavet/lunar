from enum import Enum
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# project imports
from rla2048.fts.merge import merge_left, merge_down, merge_right, merge_up


class Actions(Enum):
    left = 0
    down = 1
    right = 2
    up = 3


class Env2048(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'rgb_array_list'],
                'render_fps': 4}

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(4)
        # 4x4 grid with 16bit onehot encoding
        self.board = np.zeros((4, 4), dtype=np.uint16)
        self.score = 0
        self.observation_space = spaces.MultiBinary(256)
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def add_random_tile(self):
        r, c = random.choice(np.argwhere(self.board == 0).tolist())
        self.board[r, c] = 2 if random.random() < 0.9 else 4

    def get_obs(self):
        obs = np.zeros((4, 4, 16), dtype=np.int8)
        rs, cs = np.where(self.board != 0)
        one_hot = np.log2(self.board[rs, cs]).astype(np.int8)
        obs[rs, cs, one_hot] = 1
        return obs.flatten()

    def get_info(self):
        return {'score': self.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.uint16)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        new_board, reward = self.action_to_merge(action)
        if not np.array_equal(self.board, new_board):
            self.add_random_tile()
            self.board = new_board

        self.score += reward
        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, self.game_over, False, info

    def action_to_merge(self, action):
        if action == 0:
            new_board, reward = merge_left(self.board)
        elif action == 1:
            new_board, reward = merge_down(self.board)
        elif action == 2:
            new_board, reward = merge_right(self.board)
        elif action == 3:
            new_board, reward = merge_up(self.board)
        else:
            raise ValueError('invalid action')
        return new_board, reward

    @property
    def game_over(self) -> bool:
        if np.any(self.board == 0):
            return True
        for action in Actions:
            new_board, _ = self.action_to_merge(action.value)
            if not np.array_equal(self.board, new_board):
                return True
        return False


    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
