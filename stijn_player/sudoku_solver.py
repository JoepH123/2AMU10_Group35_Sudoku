import time

class SudokuSolver:
    """
    Our own sudoku solver implementation. This solver uses the assumption that the sudoku is not yet violated and is still solvable, as this would otherwise be recognized by the Oracle. 
    The goal of this is to compute a completed sudoku once every round. This provides two benefits:

    1. We don't need to try different numbers in the same cell, therefore the number of possible moves strongly decreases. Since the number, as long as they are valid,
       don't really matter, we might as well just have a single correct number for each cell, which are used during the minimax 'simulation'. Since there are less possible
       moves, there are less branches in minimax, making it more efficient.
    2. We can never make any taboo moves which can lead to us missing a turn, or missing out on points. 
    """

    def __init__(self, board, N, m, n):
        if len(board) != N * N:
            raise ValueError(f"Incorrect board size: Expected {N * N} cells, got {len(board)}.")
        if m * n != N:
            raise ValueError(f"Incorrect block dimensions: {m}x{n} must equal {N}.")
        self.N = N  # Size of the board (N x N)
        self.m = m  # Number of rows in a block
        self.n = n  # Number of columns in a block
        self.board = board[:]  # Make a copy of the board
        self.cells = [(i, j) for i in range(N) for j in range(N)]
        # Initialize the rows, columns, and blocks --> we could also use lists, with their index, but dictionary has clearer key-value structure)
        self.rows = {row_index: set() for row_index in range(N)}
        self.cols = {col_index: set() for col_index in range(N)}
        self.blocks = {block_index: set() for block_index in range(N)}  # The board is always made up of N blocks (e.g. (N/m) * (N/n) = n * m = N)
        self._initialize()

    def _initialize(self):
        """
        Initialize the sets for rows, columns, and blocks based on the initial board. 
        """
        for index, value in enumerate(self.board):  # go through all cells of the board
            if value != 0:  # if the value is not zero
                # We maintain dictionaries for the rows, columns and blocks. They all have indices and values 
                row, col = divmod(index, self.N)
                block = self._get_block(row, col)
                # Add current values of the board to the sets of the correct rows, column and block
                self.rows[row].add(value) 
                self.cols[col].add(value)
                self.blocks[block].add(value)

    def _get_block(self, row, col):
        """
        (row // self.m) determines the block row index.
        (self.N // self.n) gives the number of blocks per block row.
        (col // self.n) determines the block column index.
        Multiplying the block row index by the number of blocks per block row and adding the block column index gives the correct block index.

        Very logical procedure analogous to flattening rows in a matrix: 
        When flattening out the cells of a matrix into a single list. By concatenating the rows and finding out 
        what index the original cell with position i, j has in the new flattened list. Now instead of cells we reason with blocks of cells. 
        """
        return (row // self.m) * (self.N // self.n) + (col // self.n)

    def _find_empty_cell(self):
        """
        Find the next empty cell using the MRV heuristic. The MRV heuristic speeds up the sudoku solver, 
        by selected the cell in which the least amount of numbers can be placed. 
        """
        min_candidates = self.N + 1
        target_cell = None
        for row in range(self.N):
            for col in range(self.N):
                # Given that the board is represented by a single long list, the cell i, j is found in position i * N + j 
                if self.board[row * self.N + col] == 0:  
                    candidates = self._get_candidates(row, col)
                    num_candidates = len(candidates)
                    if num_candidates == 0:
                        return None  # No solution possible
                    if num_candidates < min_candidates:
                        # Find cell with    fewest candidates possible.
                        min_candidates = num_candidates
                        target_cell = (row, col)
                        if min_candidates == 1:
                            # Can't do better than this, if only one candidate this there is no cell with fewer candidates
                            return target_cell
        return target_cell

    def _get_candidates(self, row, col):
        """Get possible candidates for a given cell, candidates are the numbers that can be placed in the cell"""
        block = self._get_block(row, col)
        used = self.rows[row] | self.cols[col] | self.blocks[block]
        return [num for num in range(1, self.N + 1) if num not in used]

    def solve(self):
        """Solve the Sudoku puzzle using backtracking"""
        cell = self._find_empty_cell()
        if not cell:
            # Either solved or no solution exists
            return all(cell_val != 0 for cell_val in self.board)  # return True if all cells are non-zero (sudoku is solved) otherwise return False (sudoku is unsolvable)

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
                # Recursive call to the function in if statement, to make sure it only stops once the function returns True once (when it is solved). 
                # This happens via all(cell_val != 0 for cell_val in self.board) above.
                return True  # Solved

            # Backtrack --> if the number (num) in the candidates fails (it causes an unsolvable sudoku), False is returns in the "if self.solve():" 
            # and thus is backtracking removes this number from the cell. The next number (num) in the candidates is then attempted.
            self.board[index] = 0
            self.rows[row].remove(num)
            self.cols[col].remove(num)
            self.blocks[block].remove(num)

        return False  # Sudoku is unsolvable --> all candidates failed

    def get_board_as_dict(self):
        """Return the solved board as a dictionary with (i, j) as keys and the solved numbers as values"""
        if not self.solve():
            raise ValueError("Sudoku no longer solvable (should be notified by the Oracle, so in practice does not occur)")
        return {(i, j): self.board[i * self.N + j] for i in range(self.N) for j in range(self.N)}
    

    def print_board(self):
        """Print the solved Sudoku --> just for visual purposes"""
        for i in range(self.N):
            row = self.board[i*self.N:(i+1)*self.N]
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
