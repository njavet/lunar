import numpy as np


def merge_left(board: np.ndarray) -> tuple[np.ndarray, float]:

    def _merge_row(row: list, acc: list, r: float = 0) -> tuple[list[int], float]:
        if not row:
            return acc, r
        x = row[0]
        if len(row) == 1:
            return acc + [x], r
        if x == row[1]:
            new_row = row[2:]
            new_acc = acc + [2 * x]
            new_r = r + 2 * x
            return _merge_row(new_row, new_acc, new_r)
        else:
            new_row = row[1:]
            new_acc = acc + [x]
            return _merge_row(new_row, new_acc, r)

    new_board = []
    reward = 0
    for row_ in board:
        shifted_row = row_[row_ != 0].tolist()
        merged_row, r_ = _merge_row(shifted_row, acc=[])
        padded_row = np.pad(merged_row, (0, 4 - len(merged_row)))
        new_board.append(padded_row)
        reward += r_
    return np.array(new_board, dtype=np.uint16), reward


def merge_right(board):
    new_board = np.fliplr(board)
    new_board, reward = merge_left(new_board)
    new_board = np.fliplr(new_board)
    return new_board, reward


def merge_down(board):
    new_board = np.rot90(board, -1)
    new_board, reward = merge_left(new_board)
    new_board = np.rot90(new_board)
    return new_board, reward


def merge_up(board):
    new_board = np.rot90(board)
    new_board, reward = merge_left(new_board)
    new_board = np.rot90(new_board, -1)
    return new_board, reward
