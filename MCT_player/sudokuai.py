import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .MCT_functions import *
from .roll_out_functions import *
from .sudoku_solver import SudokuSolver

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):

    def __init__(self):
        """Initialize the SudokuAI by calling the superclass initializer."""
        super().__init__()

    def backpropagate(self, node, result):
        """
        Traverse up the tree from 'node' to the root, updating the statistics.
        'result': outcome from the perspective of the final state's player
        'root_player': the player (1 or -1) who was the current_player at the root
        """
        # If the result is from the perspective of the final move's player,
        # we need to convert it to the perspective of root_player.
        # If root_player == 1, the result is fine. If root_player == -1, we invert the sign.
        # Alternatively, you can keep track of the sign by comparing node.state.current_player
        # with root_player. For simplicity, let's interpret 'result' from the viewpoint of Player1,
        # and then invert if root_player == -1.

        if node.my_player == 2:
            # flip 1 <-> -1
            result = -result

        current_node = node
        while current_node is not None:
            current_node.update(result)
            # Because after each move, the player flips, we should also flip 'result'
            # to correctly attribute wins/losses up the tree.
            result = -result
            current_node = current_node.parent



    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute and propose the best move for the current game state using iterative deepening and minimax algorithm.

        Parameters:
            game_state (GameState): The current state of the game.

        Returns:
            None: Proposes the best move directly using self.propose_move.
        """

        solver = SudokuSolver(game_state.board.squares, game_state.board.N, game_state.board.m, game_state.board.n)
        solved_board_dict = solver.get_board_as_dict()

        root_node = MCT_node(game_state)

        for _ in range(1000):
            # 1. SELECTION: start at root node and select until leaf
            node = root_node
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # 2. EXPANSION: if not terminal, expand
            if not node.is_terminal():
                node = node.expand()
                if node is None:
                    # No moves to expand; continue
                    continue

            # 3. SIMULATION (ROLL-OUT)
            t0 = time.time()
            result = rollout(node)
            #print(time.time() - t0)

            # 4. BACKPROPAGATION
            self.backpropagate(node, result)

            # After iterations, pick the child with the highest win rate
            best_move = None
            best_ratio = -float('inf')
            for child in root_node.children:
                if child.visits > 0:
                    win_ratio = child.wins / child.visits
                    if win_ratio > best_ratio:
                        best_ratio = win_ratio
                        best_move = child.move
            self.propose_move(Move(best_move, solved_board_dict[best_move]))
            print(_)