from enum import Enum
import random
import numpy as np


class Actions(Enum):
    left = 0
    down = 1
    right = 2
    up = 3


def merge_left(board: np.ndarray) -> tuple[np.ndarray, float]:

    def _merge_row(row: list, acc: list, s: float = 0) -> tuple[list[int], float]:
        if not row:
            return acc, s
        x = row[0]
        if len(row) == 1:
            return acc + [x], s
        if x == row[1]:
            new_row = row[2:]
            new_acc = acc + [2 * x]
            new_r = s + 2 * x
            return _merge_row(new_row, new_acc, new_r)
        else:
            new_row = row[1:]
            new_acc = acc + [x]
            return _merge_row(new_row, new_acc, s)

    new_board = []
    score = 0
    for row_ in board:
        shifted_row = row_[row_ != 0].tolist()
        merged_row, s_ = _merge_row(shifted_row, acc=[])
        padded_row = np.pad(merged_row, (0, 4 - len(merged_row)))
        new_board.append(padded_row)
        score += s_
    return np.array(new_board), score


def merge_right(board: np.ndarray) -> tuple[np.ndarray, float]:
    flipped = np.fliplr(board)
    left_flipped, score = merge_left(flipped)
    new_board = np.fliplr(left_flipped)
    return new_board, score


def merge_down(board: np.ndarray) -> tuple[np.ndarray, float]:
    rotated = np.rot90(board, -1)
    left_rotated, score = merge_left(rotated)
    new_board = np.rot90(left_rotated)
    return new_board, score


def merge_up(board: np.ndarray) -> tuple[np.ndarray, float]:
    rotated = np.rot90(board)
    left_rotated, score = merge_left(rotated)
    new_board = np.rot90(left_rotated, -1)
    return new_board, score


def execute_action(board: np.ndarray, action: int) -> tuple[np.ndarray, float]:
    if action == 0:
        new_board, score = merge_left(board)
    elif action == 1:
        new_board, score = merge_down(board)
    elif action == 2:
        new_board, score = merge_right(board)
    elif action == 3:
        new_board, score = merge_up(board)
    else:
        raise ValueError(f'invalid action: {action}')
    return new_board, score


def game_over(board: np.ndarray) -> bool:
    if np.any(board == 0):
        return False
    for action in Actions:
        new_board, _ = execute_action(board, action.value)
        if not np.equal(board, new_board):
            return False
    return True


def add_random_tile(board: np.ndarray) -> np.ndarray:
    r, c = np.argwhere(board == 0)
    new_board = board.copy()
    new_board[r, c] = 2 if random.random() < 0.9 else 4
    return new_board
