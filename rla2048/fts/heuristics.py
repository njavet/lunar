import numpy as np


def max_tile_heuristic(board: np.array) -> int:
    return int(np.log2(np.max(board)))


def zero_tile_heuristic(board: np.array) -> int:
    return 16 - np.count_nonzero(board)


def smoothness_heuristic(board: np.array) -> float:
    smoothness = 0

    for row in board:
        for i in range(3):
            if row[i] > 0 and row[i + 1] > 0:
                smoothness += abs(row[i] - row[i + 1])

    for col in board.transpose():
        for i in range(3):
            if col[i] > 0 and col[i + 1] > 0:
                smoothness += abs(col[i] - col[i + 1])
    return smoothness


def monotonicity_heuristic(grid: np.ndarray) -> float:
    monotonicity = 0

    def calculate_monotonicity(array):
        increasing, decreasing = 0, 0
        for i in range(len(array) - 1):
            if array[i] >= array[i + 1]:
                decreasing += array[i] - array[i + 1]
            elif array[i] <= array[i + 1]:
                increasing += array[i + 1] - array[i]
            return increasing, decreasing

    # Calculate monotonicity for rows
    for row in grid:
        monotonicity += min(calculate_monotonicity(row))

    # Calculate monotonicity for columns
    monotonicity += calculate_monotonicity(grid.transpose()[0])[1] * 10
    for col in grid.transpose()[1:]:
        monotonicity += min(calculate_monotonicity(col))
    return monotonicity


def corner_heuristic(board: np.array) -> int:
    weights = np.array([[8, 0, 0, 0],
                        [3, -1, -1, -1],
                        [2, -1, -1, -1],
                        [1, 1, -1, -1]])
    return int(np.sum(board * weights))


def utility(self, grid: np.array) -> float:
    max_tile = self.max_tile_heuristic(grid)
    zeros = self.zero_tile_heuristic(grid)
    smooth = self.smoothness_heuristic(grid)
    mono = self.monotonicity_heuristic(grid)
    corner = self.corner_heuristic(grid)

    return mono + corner + zeros + max_tile + smooth


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
