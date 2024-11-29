import time
from typing import List, Optional, Tuple, Dict
import math

class SudokuSolver:
    def __init__(self, board: List[int], N: int, m: int, p: int):
        if len(board) != N * N:
            raise ValueError(f"Board size mismatch: Expected {N * N} cells, got {len(board)}.")
        if m * p != N:
            raise ValueError(f"Invalid block dimensions: {m}x{p} must equal {N}.")
        
        self.N = N              # Size of the board (N x N)
        self.m = m              # Number of rows in a block
        self.p = p              # Number of columns in a block
        self.board = board[:]   # Make a copy of the board
        self.cells = [(i, j) for i in range(N) for j in range(N)]
        # Precompute rows, columns, and blocks
        self.rows = [set() for _ in range(N)]
        self.cols = [set() for _ in range(N)]
        self.blocks = [set() for _ in range(N)]
        self._initialize()

    def _initialize(self):
        """Initialize the sets for rows, columns, and blocks based on the initial board."""
        for index, value in enumerate(self.board):
            if value != 0:
                row, col = divmod(index, self.N)
                block = self._get_block(row, col)
                if value in self.rows[row]:
                    raise ValueError(f"Duplicate value {value} found in row {row}.")
                if value in self.cols[col]:
                    raise ValueError(f"Duplicate value {value} found in column {col}.")
                if value in self.blocks[block]:
                    raise ValueError(f"Duplicate value {value} found in block {block}.")
                self.rows[row].add(value)
                self.cols[col].add(value)
                self.blocks[block].add(value)

    def _get_block(self, row: int, col: int) -> int:
        """Get the block index for a given cell."""
        return (row // self.m) * self.p + (col // self.p)

    def _find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Find the next empty cell using the MRV heuristic."""
        min_candidates = self.N + 1
        target_cell = None
        for row in range(self.N):
            for col in range(self.N):
                if self.board[row * self.N + col] == 0:
                    candidates = self._get_candidates(row, col)
                    num_candidates = len(candidates)
                    if num_candidates == 0:
                        return None  # No solution possible
                    if num_candidates < min_candidates:
                        min_candidates = num_candidates
                        target_cell = (row, col)
                        if min_candidates == 1:
                            return target_cell  # Can't do better than this
        return target_cell

    def _get_candidates(self, row: int, col: int) -> List[int]:
        """Get possible candidates for a given cell."""
        block = self._get_block(row, col)
        used = self.rows[row] | self.cols[col] | self.blocks[block]
        return [num for num in range(1, self.N + 1) if num not in used]

    def solve(self) -> bool:
        """Solve the Sudoku puzzle using backtracking."""
        cell = self._find_empty_cell()
        if not cell:
            # Either solved or no solution exists
            return all(cell_val != 0 for cell_val in self.board)

        row, col = cell
        index = row * self.N + col
        candidates = self._get_candidates(row, col)

        for num in candidates:
            # Place the number
            self.board[index] = num
            self.rows[row].add(num)
            self.cols[col].add(num)
            block = self._get_block(row, col)
            self.blocks[block].add(num)

            if self.solve():
                return True  # Solved

            # Backtrack
            self.board[index] = 0
            self.rows[row].remove(num)
            self.cols[col].remove(num)
            self.blocks[block].remove(num)

        return False  # Trigger backtracking

    def get_board_as_dict(self) -> Dict[Tuple[int, int], int]:
        """Return the solved board as a dictionary with (i, j) as keys and the solved numbers as values"""
        if not self.solve():
            raise ValueError("No solution exists for the given Sudoku puzzle")
        return {(i, j): self.board[i * self.N + j] for i in range(self.N) for j in range(self.N)}
    

    def print_board(self):
        """Print the solved Sudoku --> just for visual purposes"""
        for i in range(N):
            row = self.board[i*N:(i+1)*N]
            print(" ".join(str(num) for num in row))


# Test to show that it works
if __name__ == "__main__":
    # Example for standard 9x9 Sudoku with 3x3 blocks
    N = 9
    m = 3
    p = 3

    # Partly finished board example
    # board = [
    #     5, 3, 0, 0, 7, 0, 0, 0, 0,
    #     6, 0, 0, 1, 9, 5, 0, 0, 0,
    #     0, 9, 8, 0, 0, 0, 0, 6, 0,
    #     8, 0, 0, 0, 6, 0, 0, 0, 3,
    #     4, 0, 0, 8, 0, 3, 0, 0, 1,
    #     7, 0, 0, 0, 2, 0, 0, 0, 6,
    #     0, 6, 0, 0, 0, 0, 2, 8, 0,
    #     0, 0, 0, 4, 1, 9, 0, 0, 5,
    #     0, 0, 0, 0, 8, 0, 0, 7, 9
    # ]

    # Completely unfinished board example
    board = N**2 * [0]

    try:
        start = time.perf_counter()
        solver = SudokuSolver(board, N, m, p)
        solved_board_dict = solver.get_board_as_dict()
        end = time.perf_counter()
        print(f"Time for solving: {end-start}")
        solver.print_board()
    except ValueError as e:
        print(e)
