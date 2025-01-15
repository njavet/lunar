from enum import Enum
import random
import numpy as np


class Actions(Enum):
    left = 0
    down = 1
    right = 2
    up = 3


def state_to_board(state: np.ndarray, board: np.ndarray):
    board.fill(0)
    one_hots, rs, cs = np.where(state != 0)
    values = np.exp2(one_hots).astype(np.uint16)
    board[rs, cs] = values


def board_to_state(board: np.ndarray) -> np.ndarray:
    # TODO move back to log-1 (since the first grid is always zero)
    state = np.zeros((16, 4, 4), dtype=np.float32)
    rs, cs = np.where(board != 0)
    one_hot = np.log2(board[rs, cs]).astype(np.uint8)
    state[one_hot, rs, cs] = 1
    return state


def shift_left(board: np.ndarray) -> np.ndarray:
    # TODO shift and merge without converting to board
    shifted_board = np.array([np.pad(row[row != 0],
                                     (0, len(row) - np.count_nonzero(row)))
                              for row in board])
    return shifted_board


def merge_left(board: np.ndarray) -> np.ndarray:
    shifted = shift_left(board)
    merged = shifted.copy()
    mask = (shifted[:, :-1] == shifted[:, 1:]) & (shifted[:, :-1] != 0)
    merged[:, :-1][mask] *= 2
    merged[:, 1:][mask] = 0
    return shift_left(merged)


def merge_right(board: np.ndarray) -> np.ndarray:
    flipped = np.fliplr(board)
    left_flipped = merge_left(flipped)
    new_board = np.fliplr(left_flipped)
    return new_board


def merge_down(board: np.ndarray) -> np.ndarray:
    rotated = np.rot90(board, -1)
    left_rotated = merge_left(rotated)
    new_board = np.rot90(left_rotated)
    return new_board


def merge_up(board: np.ndarray) -> np.ndarray:
    rotated = np.rot90(board)
    left_rotated = merge_left(rotated)
    new_board = np.rot90(left_rotated, -1)
    return new_board


def get_score(board: np.ndarray, merged: np.ndarray) -> int:
    b0_tiles = dict(zip(*np.unique(board, return_counts=True)))
    b1_tiles = dict(zip(*np.unique(merged, return_counts=True)))
    score = 0
    for tile, count in b1_tiles.items():
        if tile not in b0_tiles:
            score += tile
        else:
            score += tile * abs(count - b0_tiles[tile])
    return score


def execute_action(board: np.ndarray, action: int) -> np.ndarray:
    if action == 0:
        new_board = merge_left(board)
    elif action == 1:
        new_board = merge_down(board)
    elif action == 2:
        new_board = merge_right(board)
    elif action == 3:
        new_board = merge_up(board)
    else:
        raise ValueError(f'invalid action: {action}')
    return new_board


def game_over(board: np.ndarray) -> bool:
    if np.any(board == 0):
        return False
    for action in Actions:
        new_board = execute_action(board, action.value)
        if not np.array_equal(board, new_board):
            return False
    return True


def add_random_tile(board):
    empty_cells = np.argwhere(board == 0)
    if empty_cells.size > 0:
        row, col = empty_cells[np.random.randint(len(empty_cells))]
        board[row, col] = np.random.choice([2, 4], p=[0.9, 0.1])
