import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box, Discrete
import numpy as np
import pygame

# project imports
from lunar.g2048 import state
from lunar import config


class Env2048(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'rgb_array_list', 'ansi'],
                'render_fps': 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.observation_space = Box(low=0, high=1, shape=(16, 4, 4))
        self.action_space = Discrete(4)
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.score = 0
        self.window_size = 512
        self.window = None
        self.clock = None
        self.corners = self.board[[0, 0, 3, 3], [0, 3, 0, 3]]

    def get_obs(self):
        return state.board_to_state(self.board)

    def get_info(self):
        return {'score': self.score,
                'tiles': dict(zip(*np.unique(self.board, return_counts=True)))}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.score = 0
        state.add_random_tile(self.board)
        state.add_random_tile(self.board)
        observation = self.get_obs()
        info = {'score': 0, 'tiles': {}}

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        new_board = state.execute_action(self.board, action)
        score = state.get_score(self.board, new_board)
        self.score += score

        if not np.array_equal(self.board, new_board):
            state.add_random_tile(new_board)
            # in_corner = np.any(self.corners == self.max_tile)
            # zeros = 16 - np.count_nonzero(self.board)
            reward = score + 1
        else:
            reward = -1

        observation = self.get_obs()
        game_over = state.game_over(self.board)
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, game_over, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'rgb_array':
            return self._render_frame()

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
        # rgb_array
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
