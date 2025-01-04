import unittest
import torch

# project imports
from rla2048.fts import merge_left, merge_down, merge_right, merge_up


class TestMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid0 = torch.tensor([[2, 2, 2, 2],
                                  [4, 4, 0, 0],
                                  [2, 4, 8, 16],
                                  [2, 0, 2, 4]])

        cls.grid1 = torch.tensor([[2, 2, 4, 2],
                                  [8, 4, 4, 16],
                                  [8, 0, 0, 16],
                                  [0, 0, 0, 32]])

    def test_merge_left_grid0(self):
        grid, _ = merge_left(self.grid0)
        grid_res = torch.tensor([[4, 4, 0, 0],
                                 [8, 0, 0, 0],
                                 [2, 4, 8, 16],
                                 [4, 4, 0, 0]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_left_score_grid0(self):
        _, score = merge_left(self.grid0)
        self.assertEqual(score, 20)

    def test_merge_left_grid1(self):
        grid, _ = merge_left(self.grid1)
        grid_res = torch.tensor([[4, 4, 2, 0],
                                 [8, 8, 16, 0],
                                 [8, 16, 0, 0],
                                 [32, 0, 0, 0]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_left_score_grid1(self):
        _, score = merge_left(self.grid1)
        self.assertEqual(score, 12)

    def test_merge_right_grid0(self):
        grid, _ = merge_right(self.grid0)
        grid_res = torch.tensor([[0, 0, 4, 4],
                                 [0, 0, 0, 8],
                                 [2, 4, 8, 16],
                                 [0, 0, 4, 4]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_right_grid1(self):
        grid, _ = merge_right(self.grid1)
        grid_res = torch.tensor([[0, 4, 4, 2],
                                 [0, 8, 8, 16],
                                 [0, 0, 8, 16],
                                 [0, 0, 0, 32]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_up_grid0(self):
        grid, _ = merge_up(self.grid0)
        grid_res = torch.tensor([[2, 2, 2, 2],
                                 [4, 8, 8, 16],
                                 [4, 0, 2, 4],
                                 [0, 0, 0, 0]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_up_score_grid0(self):
        _, score = merge_up(self.grid0)
        self.assertEqual(score, 12)

    def test_merge_up_grid1(self): 
        grid, _ = merge_up(self.grid1)
        grid_res = torch.tensor([[2, 2, 8, 2],
                                 [16, 4, 0, 32],
                                 [0, 0, 0, 32],
                                 [0, 0, 0, 0]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_up_score_grid1(self):
        _, score = merge_up(self.grid1)
        self.assertEqual(score, 56)

    def test_merge_down_grid0(self):
        grid, _ = merge_down(self.grid0)
        grid_res = torch.tensor([[0, 0, 0, 0],
                                 [2, 0, 2, 2],
                                 [4, 2, 8, 16],
                                 [4, 8, 2, 4]])
        self.assertTrue(torch.equal(grid, grid_res))

    def test_merge_down_grid1(self):
        grid, _ = merge_down(self.grid1)
        grid_res = torch.tensor([[0, 0, 0, 0],
                                 [0, 0, 0, 2],
                                 [2, 2, 0, 32],
                                 [16, 4, 8, 32]])
        self.assertTrue(torch.equal(grid, grid_res))


if __name__ == '__main__':
    unittest.main()
