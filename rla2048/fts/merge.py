import numpy as np


def merge_row_left(row: np.ndarray,
                   acc: np.ndarray,
                   score: float = 0) -> tuple[np.ndarray, float]:
    if not row:
        return acc, score
    x = row[0]
    if len(row) == 1:
        return acc + [x], score

    if x == row[1]:
        new_row = row[2:]
        new_acc = acc + [2 * x]
        new_score = score + 2 * x
        return merge_row_left(new_row, new_acc, new_score)
    else:
        new_row = row[1:]
        new_acc = acc + [x]
        new_score = score
        return merge_row_left(new_row, new_acc, new_score)


def merge_left(grid):
    new_board = []
    reward = 0
    for i, row in enumerate(grid):
        merged, r = merge_row_left([x for x in row if x != 0], [])
        zeros = len(row) - len(merged)
        merged_zeros = merged + zeros * [0]
        new_board.append(merged_zeros)
        reward += r
    return np.array(new_board, dtype=np.uint16), reward


def merge_right(board):
    new_board = np.fliplr(board)
    new_board, reward = merge_left(new_board)
    new_board = np.fliplr(new_board)
    return new_board, reward


def merge_down(board):
    new_board = np.rot90(board)
    new_board, reward = merge_left(new_board)
    new_board = np.rot90(new_board, -1)
    return new_board, reward

def merge_up(board):
    new_board = np.rot90(board, -1)
    new_board, reward = merge_left(new_board)
    new_board = np.rot90(new_board)
    return new_board, reward
