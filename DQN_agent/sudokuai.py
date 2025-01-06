#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
import pickle
import torch
torch.set_num_threads(1) 
import os
import sys

# Add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from .dqn import CNNQNetwork, CNNQNetwork_6x6
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

    def hash_key(self):
        """
        Generate a unique hash key for this node.
        """
        return hash((tuple(self.occupied_squares1), tuple(self.occupied_squares2), tuple(self.scores), self.current_player))


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


    def load_model(self, filename="dqn_model_2x2.pkl", input_shape=(9,9), num_actions=81):
        """Load the trained model from the 'models' folder in the same directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Map van dit script
        models_dir = os.path.join(script_dir, "models")  # Pad naar 'models' map
        file_path = os.path.join(models_dir, filename)  # Volledige pad naar modelbestand

        # Model laden
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Pas de input_shape en num_actions aan overeenkomstig je nieuwe architectuur
        if input_shape == (9,9):
            model = CNNQNetwork(input_shape, num_actions)
        elif input_shape == (6,6):
            model = CNNQNetwork_6x6(input_shape, num_actions)

        model.load_state_dict(data["policy_state_dict"])
        model.eval()
        print(f"Model loaded from {file_path}")
        return model
    

    def preprocess_board(self, original_board, shape=(9,9)):
        # Convert squares to a 9x9 numpy array
        squares = original_board.squares.copy()
        board_array = np.array(squares).reshape(shape)

        # Mark squares owned by player 1 as 1
        board_array[board_array > 0] = 1

        # Mark squares owned by player 2 as -1
        board_array[board_array < 0] = -1

        # Zeros remain 0 (empty)
        return board_array


    def make_state(self, board, player=1):
        p1_channel = (board == player).astype(np.float32)   
        p2_channel = (board == -player).astype(np.float32)  
        empty_channel = (board == 0).astype(np.float32)     
        
        state = np.stack([p1_channel, p2_channel, empty_channel], axis=0)
        return state


    def select_max_q_action(self, model, state, valid_moves, N=9):
        """Select the action with the highest Q-value for a 9x9 board."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (1, 3, 9, 9)
        with torch.no_grad():
            q_values = model(state_t).squeeze(0).numpy()  # (81,)
        
        # Mask invalid moves
        q_values_masked = np.full_like(q_values, -1e9)
        for r, c in valid_moves:
            action_idx = r * N + c  # Updated for 9x9
            q_values_masked[action_idx] = q_values[action_idx]
        
        best_action_idx = np.argmax(q_values_masked)
        return best_action_idx // N, best_action_idx % N  # Updated for 9x9


    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute and propose the best move for the current game state using iterative deepening and minimax algorithm.

        Parameters:
            game_state (GameState): The current state of the game.

        Returns:
            None: Proposes the best move directly using self.propose_move.
        """

        # Initialize the root node for the game state
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player

        # Reset the nodes explored counter for this computation
        self.nodes_explored = 0

        # Retrieve all possible moves for the current state
        all_moves = self.get_all_moves(root_node)

        if (game_state.board.n, game_state.board.m) == (3,3): 
            print('loading dqn model for 9x9 board')
            model = self.load_model(filename='team35_9x9_dqn_model.pkl')
            squares = [move.square for move in all_moves]
            current_board = self.preprocess_board(game_state.board)
            state = self.make_state(current_board, player=root_node.my_player)
            action = self.select_max_q_action(model, state, squares)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return
        
        elif (game_state.board.n, game_state.board.m) == (3, 2):
            print('loading dqn model for 6x6 board')
            model = self.load_model(filename='team35_6x6_dqn_model.pkl', input_shape=(6,6), num_actions=36)
            squares = [move.square for move in all_moves]
            current_board = self.preprocess_board(game_state.board, shape=(6,6))
            state = self.make_state(current_board, player=root_node.my_player)
            action = self.select_max_q_action(model, state, squares, N=6)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return

        else:
            # Randomly propose a move initially to ensure there's always a fallback
            self.propose_move(random.choice(all_moves))

