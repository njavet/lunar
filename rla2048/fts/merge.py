import torch
import numpy as np


def merge_left(board: torch.Tensor) -> tuple[torch.Tensor, float]:

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
    return torch.tensor(np.array(new_board), device='cuda'), score


def merge_right(board: torch.Tensor) -> tuple[torch.Tensor, float]:
    flipped = torch.fliplr(board)
    left_flipped, score = merge_left(flipped)
    new_board = torch.fliplr(left_flipped)
    return new_board, score


def merge_down(board: torch.Tensor) -> tuple[torch.Tensor, float]:
    rotated = torch.rot90(board, -1)
    left_rotated, score = merge_left(rotated)
    new_board = torch.rot90(left_rotated)
    return new_board, score


def merge_up(board: torch.Tensor) -> tuple[torch.Tensor, float]:
    rotated = torch.rot90(board)
    left_rotated, score = merge_left(rotated)
    new_board = torch.rot90(left_rotated, -1)
    return new_board, score
