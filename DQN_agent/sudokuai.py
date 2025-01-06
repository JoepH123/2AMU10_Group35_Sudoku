#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

# onnx-runtime
import onnxruntime as ort
import numpy as np

# Add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .sudoku_solver import SudokuSolver


class NodeGameState(GameState):
    def __init__(self, game_state, root_move=None, last_move=None, my_player=None):
        """
        Initialize a NodeGameState by copying the given game_state and adding extra attributes.

        Parameters:
        - game_state (GameState): The game state to copy.
        - root_move (Move, optional): The initial move leading to this node.
        - last_move (Move, optional): The last move made.
        - my_player (int, optional): The AI's player number.
        """
        self.__dict__ = game_state.__dict__.copy()
        self.root_move = root_move
        self.last_move = last_move
        self.my_player = my_player
        # Given input board find valid values for all cells, using a sudoku solver
        solver = SudokuSolver(game_state.board.squares, game_state.board.N, game_state.board.m, game_state.board.n)
        solved_board_dict = solver.get_board_as_dict()
        self.solved_board_dict = solved_board_dict


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    def __init__(self):
        """Initialize the SudokuAI by calling the superclass initializer."""
        super().__init__()

    def get_all_moves(self, node):
        """
        Generate all possible valid moves for the current player.

        Parameters:
        - node (NodeGameState): The current game state.

        Returns:
        - list of Move: A list of all valid moves.
        """
        player_squares = node.player_squares()
        all_moves = [Move(coordinates, node.solved_board_dict[coordinates]) for coordinates in player_squares]
        return all_moves

    # --- 1) Nieuwe functie: ONNX-model laden zonder PyTorch ---
    def load_onnx_model(self, filename="team35_9x9_dqn_model.onnx"):
        """
        Loads a ONNX-model via onnxruntime.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        file_path = os.path.join(models_dir, filename)

        # Initialiseer een onnxruntime sessie (CPU-only)
        session = ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
        print(f"ONNX-model loaded from {file_path}")
        return session

    def preprocess_board(self, original_board, shape=(9,9)):
        board_array = np.asarray(original_board.squares).reshape(shape)
        board_array = np.where(board_array > 0, 1, np.where(board_array < 0, -1, 0))
        return board_array

    def make_state(self, board, player=1):
        """
        Making a 3-channel representation:
          kanaal 0 = player cells
          kanaal 1 = oppponent cells
          kanaal 2 = empty cells
        """
        p1_channel = (board == player).astype(np.float32)
        p2_channel = (board == -player).astype(np.float32)
        empty_channel = (board == 0).astype(np.float32)

        state = np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # shape (3,9,9)
        return state

    # --- 2) Actieselectie met ONNX-runtime ---
    def select_max_q_action(self, onnx_session, state, valid_moves, N=9):
        """
        Calculates Q-values via onnxruntime and chooses best legal move.
        - onnx_session: onnxruntime.InferenceSession
        - state: numpy-array shape (3,9,9)
        - valid_moves: list with (r,c) moves
        - N: 9 (or 6 for 6x6)
        """
        # onnxruntime verwacht (batch_size, 3, 9, 9) -> dus voeg batch-dim toe
        input_data = state[np.newaxis, :].astype(np.float32)  # shape (1,3,9,9)

        # Haal input en output-namen op uit het ONNX-model
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        # Run inference
        outputs = onnx_session.run([output_name], {input_name: input_data})
        # outputs[0] is shape (1,81)
        q_values = outputs[0].squeeze(0)  # nu shape (81,)

        # Mask invalid moves
        mask = np.full(q_values.shape, -1e9, dtype=np.float32)
        for (r, c) in valid_moves:
            action_idx = r * N + c
            mask[action_idx] = q_values[action_idx]

        best_action_idx = mask.argmax()
        return best_action_idx // N, best_action_idx % N

    # --- 3) compute_best_move met ONNX ---
    def compute_best_move(self, game_state: GameState) -> None:
        # Initialize the root node for the game state
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player

        # Retrieve all possible moves
        all_moves = self.get_all_moves(root_node)
        squares = [move.square for move in all_moves]

        if (game_state.board.n, game_state.board.m) == (3,3):
            print("Loading ONNX model for 9x9 board")
            # 1. Laad ONNX-sessie
            onnx_session = self.load_onnx_model(filename="team35_9x9_dqn_model.onnx")

            # 2. Preprocess board
            current_board = self.preprocess_board(game_state.board, shape=(9,9))
            state = self.make_state(current_board, player=root_node.my_player)

            # 3. Kies beste zet
            action = self.select_max_q_action(onnx_session, state, squares, N=9)

            # 4. Propose move
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return

        elif (game_state.board.n, game_state.board.m) == (3,2):
            print("Loading ONNX model for 6x6 board")
            onnx_session = self.load_onnx_model(filename="team35_6x6_dqn_model.onnx")

            current_board = self.preprocess_board(game_state.board, shape=(6,6))
            state = self.make_state(current_board, player=root_node.my_player)

            action = self.select_max_q_action(onnx_session, state, squares, N=6)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return
        else:
            # Fallback: random move
            self.propose_move(random.choice(all_moves))


