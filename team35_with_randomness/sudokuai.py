import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import team35_current_best.sudokuai


class SudokuAI(team35_current_best.sudokuai.SudokuAI):
    """
    This Sudoku AI uses our current best approach with some percentage of randomness. This is used to evaluate our method.
    """

    def __init__(self):
        super().__init__()
        self.proportion_of_random_moves = 0.2 

    def compute_best_move(self, game_state: GameState) -> None:
        if random.random() < self.proportion_of_random_moves:  # perform random move
            print("@@ RANDOM MOVE @@")
            self.compute_random_move(game_state)
        else:
            print("@@ NORMAL MOVE @@")
            super().compute_best_move(game_state)

    def compute_random_move(self, game_state: GameState):
        all_moves = super().get_all_moves(game_state)
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))
