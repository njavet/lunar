from collections import defaultdict
import numpy as np


    def find_best_move(self, board: list[list[int]]) -> int:
        self.state = np.array(board, dtype=np.int16)
        result = np.array([self.score_top_level_move(i) for i in range(4)])

        if np.max(result) == 0:
            move = int(np.random.choice([0, 1, 2, 3]))
        else:
            move = random_argmax(result)
        return move

    def score_top_level_move(self, move):
        self.simulator.simulate_move(self.state, move)
        if np.all(self.simulator.new_state == self.state):
            return 0

        return self.expectimax(self.simulator.new_state, self.depth, agent_play=False)

    def expectimax(self, grid: np.ndarray, depth: int, agent_play: bool) -> float:
        if depth == 0:
            return self.utility(grid)

        if agent_play:
            alpha = 0
            for move in range(4):
                self.simulator.simulate_move(grid, move)
                if not np.equal(self.simulator.new_state, grid).all():
                    alpha = max(alpha, self.expectimax(self.simulator.new_state,
                                                       depth-1,
                                                       False))
            return alpha
        else:
            expected_value = 0
            zero_cells = [(i, j) for i, row in enumerate(grid)
                          for j, val in enumerate(row) if val == 0]
            zeros = len(zero_cells)

            for i, j in zero_cells:
                ng2 = copy.deepcopy(grid)
                ng2[i][j] = 2
                expected_value += 0.9 * self.expectimax(ng2, depth-1, True)

                ng4 = copy.deepcopy(grid)
                ng4[i][j] = 4
                expected_value += 0.1 * self.expectimax(ng4, depth-1, True)
            return (1 / zeros) * expected_value

@staticmethod
def max_tile_heuristic(grid: np.array):
    return int(np.log2(np.max(grid)))

@staticmethod
def zero_tile_heuristic(grid: np.array):
    return 16 - np.count_nonzero(grid)

@staticmethod
def smoothness_heuristic(grid: np.array):
        smoothness = 0

    for row in grid:
        for i in range(len(row) - 1):
            if row[i] > 0 and row[i + 1] > 0:
                smoothness += abs(row[i] - row[i + 1])

    for col in grid.transpose():
        for i in range(len(col) - 1):
            if col[i] > 0 and col[i + 1] > 0:
                smoothness += abs(col[i] - col[i + 1])
    return smoothness

@staticmethod
def monotonicity_heuristic(grid: np.ndarray):
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

@staticmethod
def corner_heuristic(grid: np.array):
    weights = np.array([[8, 0, 0, 0],
                        [3, -1, -1, -1],
                        [2, -1, -1, -1],
                        [1, 1, -1, -1]])
    return int(np.sum(grid * weights))

def utility(self, grid: np.array) -> float:
    max_tile = self.max_tile_heuristic(grid)
    zeros = self.zero_tile_heuristic(grid)
    smooth = self.smoothness_heuristic(grid)
    mono = self.monotonicity_heuristic(grid)
    corner = self.corner_heuristic(grid)

    print('maxtile', max_tile,
          'zeros', zeros,
          'smooth', smooth,
          'mono', mono,
          'corner', corner)
    return mono + corner + zeros + max_tile
