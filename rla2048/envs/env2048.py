from enum import Enum
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.core import RenderFrame

# project imports
from rla2048.fts import merge_left, merge_down, merge_right, merge_up
from rla2048.fts.heuristics import old_utility
from rla2048 import config


class Actions(Enum):
    left = 0
    down = 1
    right = 2
    up = 3


class Env2048(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'rgb_array_list', 'ansi'],
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
        self.score += reward
        if not np.array_equal(self.board, new_board):
            ut = old_utility(self.board)
            ut_new = old_utility(new_board)
            self.board = new_board
            self.add_random_tile()
            delta = ut_new - ut
            reward += 0.5
            reward += 18*delta
        else:
            # punish nop actions
            reward = -2

        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()
        if self.render_mode == 'ansi':
            self.render_ansi()

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
            raise ValueError(f'invalid action: {action}')
        return new_board, reward

    @property
    def game_over(self) -> bool:
        if np.any(self.board == 0):
            return False
        for action in Actions:
            new_board, _ = self.action_to_merge(action.value)
            if not np.array_equal(self.board, new_board):
                return False
        return True

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def render_ansi(self):
        for i, row in enumerate(self.board):
            r = ' | '.join([str(cell) for cell in row])
            print(r)

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('risktec 2048 rl agent')
            self.window = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((config.WIDTH, config.HEIGHT))
        canvas.fill((255, 255, 255))
        pygame.font.init()
        font = pygame.font.Font(None, 55)
        text_surface = font.render('Score: ' + str(self.score),
                                   True,
                                   config.FONT_COLOR)
        text_rect = text_surface.get_rect(center=(100, 50))
        canvas.blit(text_surface, text_rect)

        cell_size = config.TILE_SIZE + config.GAP_SIZE
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                cell = int(cell)
                color = config.TILE_COLORS.get(cell, config.TILE_COLORS[4096])
                # different coordinate system
                x = config.GAP_SIZE + j * cell_size + 2
                y = config.GAP_SIZE + i * cell_size + 200
                rect = pygame.Rect(x, y, config.TILE_SIZE, config.TILE_SIZE)
                pygame.draw.rect(canvas,
                                 color=color,
                                 rect=rect,
                                 border_radius=10)
                if cell != 0:
                    text_surface = font.render(str(cell),
                                               True,
                                               config.FONT_COLOR)
                    text_rect = text_surface.get_rect(center=(
                        x + config.TILE_SIZE // 2,
                        y + config.TILE_SIZE // 2))
                    canvas.blit(text_surface, text_rect)

        for x in range(5):
            pygame.draw.line(
                canvas,
                0,
                (4, cell_size * x + 204),
                (config.WIDTH - 24, cell_size * x + 204),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (cell_size * x + 4, 204),
                (cell_size * x + 4, config.WIDTH + 180),
                width=3,
            )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas`
            # to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            button_color = (0, 200, 0)
            button_rect = pygame.Rect(200, 10, 10, 250)
            pygame.draw.rect(canvas, color=button_color, rect=button_rect)
            paused = True
            self.clock.tick(self.metadata['render_fps'])
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        paused = False

            # We need to ensure that human-rendering occurs at the
            # predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
