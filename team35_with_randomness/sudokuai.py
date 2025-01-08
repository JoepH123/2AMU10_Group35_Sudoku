import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
# Change to agent you want to randomize:
import team35_A2_plus.sudokuai as agent


class SudokuAI(agent.SudokuAI):
    """
    This Sudoku AI uses some input agent with some probability of randomness. 
    This is used to evaluate our agent for robustness.
    """

    def __init__(self):
        super().__init__()
        self.proportion_of_random_moves = 0.1 # probability of random moves: 0.1 and 0.2

    def compute_best_move(self, game_state: GameState) -> None:
        if random.random() < self.proportion_of_random_moves:  # Perform random move
            print("@@ RANDOM MOVE @@")
            self.compute_random_move(game_state)
        else:  # Perform normal move
            print("@@ NORMAL MOVE @@")
            super().compute_best_move(game_state)

    def compute_random_move(self, game_state: GameState):
        """Pick a random move from all proposed moves"""
        all_moves = self.get_all_moves(game_state)
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

    def respects_rule_C0(self, node, row, col, value):
        """
        Check whether the move respects the C0 rule:
        Sudoku property: C0 The entries of non-empty cells within each region must be unique.

        - Every row does not contain the same value more than once
        - Every column does not contain the same value more than once
        - Every block does not contain the same value more than once
        """
        board = node.board
        N = node.board.N  # Size of the grid (N = n * m)
        n, m = node.board.n, node.board.m  # Block dimensions

        # Precompute block starting indices
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n

        # Check row, column, and block in a single pass
        for i in range(N):
            # Check the row
            if board.get((row, i)) == value:
                return False

            # Check the column
            if board.get((i, col)) == value:
                return False

            # Check the block
            block_row = block_start_row + i // n
            block_col = block_start_col + i % n
            if board.get((block_row, block_col)) == value:
                return False

        return True

    def playable_moves(self, node, i, j, value):
        """
        Which moves are playable:
        
        - The are empty cells
        - Not in taboo moves
        - Adjacent to player occupied cells or in starting region
        """
        return node.board.get((i, j)) == SudokuBoard.empty \
            and not TabooMove((i, j), value) in node.taboo_moves \
            and (i, j) in node.player_squares()

    def get_all_moves(self, node):
        """
        Find all legal moves:

        - Moves that are playable (method playable_moves())
        - Moves that do not violate the C0 rule of the game (method respect_rule_c0())
        """
        N = node.board.N
        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if self.playable_moves(node, i, j, value) 
                     and self.respects_rule_C0(node, i, j, value)]
        return all_moves
