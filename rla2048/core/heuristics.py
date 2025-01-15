import numpy as np
import torch


def max_tile_heuristic(board: torch.Tensor) -> float:
    return torch.max(board).item()


def zero_tile_heuristic(board: torch.Tensor) -> float:
    tmp = torch.count_nonzero(board)
    return tmp.item()


def smoothness_heuristic(board: torch.Tensor) -> float:
    distances = 0
    for row in board:
        for i in range(3):
            if row[i] > 0 and row[i + 1] > 0:
                distances += abs(row[i] - row[i + 1])
    for col in board.transpose(4, 4):
        for i in range(3):
            if col[i] > 0 and col[i + 1] > 0:
                distances += abs(col[i] - col[i + 1])
    if distances > 0:
        return 1 / distances
    else:
        return 1


def monotonicity_heuristic(board: torch.Tensor) -> float:
    monotonicity = 0

    def calculate_inc(array):
        if torch.all(array[:-1] <= array[1:]):
            return 1
        else:
            return 0

    def calculate_dec(array):
        if torch.all(array[:-1] >= array[1:]):
            return 1
        else:
            return 0

    for row in board:
        monotonicity += calculate_inc(row)
        monotonicity += calculate_dec(row)
    for col in board.transpose(0, 1):
        monotonicity += calculate_inc(col)
        monotonicity += calculate_dec(col)
    if monotonicity > 0:
        return -1 / monotonicity
    else:
        return -1


def corner_heuristic(board: torch.Tensor) -> float:
    weights = torch.tensor([[1, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]], device='cuda')
    cs = torch.sum(board * weights)
    return cs.item()


def utility(grid: torch.Tensor, weights: torch.Tensor = None) -> float:
    if weights is None:
        weights = torch.ones(5, device='cuda')
    max_tile = max_tile_heuristic(grid)
    zeros = zero_tile_heuristic(grid)
    # smooth = smoothness_heuristic(grid)
    smooth = 0
    mono = monotonicity_heuristic(grid)
    corner = corner_heuristic(grid)
    res = weights * torch.tensor([max_tile, zeros, smooth, mono, corner], device='cuda')
    return torch.sum(res).item()


def old_utility(board: torch.Tensor) -> float:
    def helper(seq: torch.Tensor) -> float:
        # number of zeros heuristic
        zeros = np.sum(seq == 0)
        # higher tiles are better
        rank = np.max(seq)
        if rank == 0:
            rw = 1
        else:
            rw = 1 / rank

        # large tiles on the edge
        ind = np.where(seq == rank)[0][0]
        if ind == 0 or ind == 3:
            edge = 1 - rw
        else:
            edge = 0

        # monotonous
        mono = 0
        mon_inc = np.all([val <= seq[i + 1] for i, val in enumerate(seq[:-1])])
        mon_dec = np.all([seq[i + 1] <= val for i, val in enumerate(seq[:-1])])
        if mon_inc:
            mono += 2
        if mon_dec:
            mono += 2

        adj = 0
        for i, val in enumerate(seq[1:]):
            if np.all(val == seq[i + 1]):
                adj += 1 - rw

        return zeros + edge + mono + adj

    tmp = np.sum([helper(board[:, i]) for i in range(4)])
    return tmp + np.sum([helper(board[i, :]) for i in range(4)])
