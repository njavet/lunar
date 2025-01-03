import numpy as np


def max_tile_heuristic(board: np.array) -> float:
    return -1 / np.max(board)


def zero_tile_heuristic(board: np.array) -> float:
    return np.count_nonzero(board) / 16


def smoothness_heuristic(board: np.array) -> float:
    distances = 0
    for row in board:
        for i in range(3):
            if row[i] > 0 and row[i + 1] > 0:
                distances += abs(row[i] - row[i + 1])
    for col in board.transpose():
        for i in range(3):
            if col[i] > 0 and col[i + 1] > 0:
                distances += abs(col[i] - col[i + 1])
    if distances > 0:
        return 1 / distances
    else:
        return 1


def monotonicity_heuristic(board: np.ndarray) -> float:
    monotonicity = 0

    def calculate_inc(array):
        if np.all([array[i] <= array[i + 1] for i in range(len(array) - 1)]):
            return 1
        else:
            return 0

    def calculate_dec(array):
        if np.all([array[i] >= array[i + 1] for i in range(len(array) - 1)]):
            return 1
        else:
            return 0

    for row in board:
        monotonicity += calculate_inc(row)
        monotonicity += calculate_dec(row)
    for col in board.transpose():
        monotonicity += calculate_inc(col)
        monotonicity += calculate_dec(col)
    if monotonicity > 0:
        return -1 / monotonicity
    else:
        return -1


def corner_heuristic(board: np.array) -> float:
    weights = np.array([[1, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 1]])
    cs = np.sum(board * weights)
    if cs > 0:
        return -1 / cs
    else:
        return -1


def utility(grid: np.ndarray, weights: np.ndarray = None) -> float:
    if weights is None:
        weights = np.ones(5)
    max_tile = max_tile_heuristic(grid)
    zeros = zero_tile_heuristic(grid)
    smooth = smoothness_heuristic(grid)
    mono = monotonicity_heuristic(grid)
    corner = corner_heuristic(grid)
    res = weights * np.array([max_tile, zeros, smooth, mono, corner])
    return np.sum(res)


def old_utility(board: np.array) -> float:
    def helper(seq: np.ndarray) -> float:
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
