import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium.spaces import MultiBinary, Discrete
import numpy as np
import pygame

# project imports
from rla2048.fts import merge, heuristics
from rla2048 import config


class Env2048(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__(observation_space=MultiBinary(256),
                         action_space=Discrete(4),
                         render_mode=render_mode)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.window_size = 512
        self.window = None
        self.clock = None

    def get_obs(self):
        obs = np.zeros((4, 4, 16), dtype=np.uint8)
        rs, cs = np.argwhere(self.board != 0)
        one_hot = np.log2(self.board[rs, cs]).astype(np.uint8)
        obs[rs, cs, one_hot] = 1
        return obs.flatten()

    def get_info(self):
        return {'score': self.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.board = merge.add_random_tile(self.board)
        self.board = merge.add_random_tile(self.board)
        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        new_board, score = merge.execute_action(self.board, action)
        self.score += score

        if not torch.equal(self.board, new_board):

            cs = heuristics.corner_heuristic(self.board)
            mt = heuristics.max_tile_heuristic(self.board)
            zh = heuristics.zero_tile_heuristic(self.board)
            self.board = merge.add_random_tile(new_board)
            reward = score + 1 + cs + mt + zh
        else:
            reward = -1

        observation = self.get_obs()
        reward = torch.tensor(reward, device=self.device)
        game_over = merge.game_over(self.board)
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
