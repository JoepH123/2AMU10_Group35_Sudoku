import random
import time
import copy
import pickle
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
from joblib import load
import onnxruntime as ort

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

    

    def load_model(self, filename="team35_9x9_dqn_model.pkl"):
        """Load the trained model from the 'models' folder in the same directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        file_path = os.path.join(models_dir, filename)
        
        # load model
        model_data = load(file_path)  # Sneller dan pickle.load
        model = CNNQNetwork()
        model.load_state_dict(model_data["policy_state_dict"])
        model.eval()
        return model
    

    def load_torchscript_model(self, filename="team35_9x9_dqn_scripted.pt"):
        """
        Laad een TorchScript-model (gecompileerd via torch.jit.trace of torch.jit.script).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        file_path = os.path.join(models_dir, filename)

        # Gebruik torch.jit.load om het TorchScript-model in te laden
        model = torch.jit.load(file_path)
        model.eval()
        print(f"TorchScript-model geladen vanaf {file_path}")
        return model

    

    def preprocess_board(self, original_board, shape=(9,9)):
        board_array = np.asarray(original_board.squares).reshape(shape)
        board_array = np.where(board_array > 0, 1, np.where(board_array < 0, -1, 0))

        return board_array


    def make_state(self, board, player=1):
        p1_channel = (board == player).astype(np.float32)   
        p2_channel = (board == -player).astype(np.float32)  
        empty_channel = (board == 0).astype(np.float32)     
        
        state = np.stack([p1_channel, p2_channel, empty_channel], axis=0)
        return state


    # def select_max_q_action(self, model, state, valid_moves, N=9):
    #     """Select the action with the highest Q-value for a 9x9 board."""
    #     state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (1, 3, 9, 9)
    #     with torch.no_grad():
    #         q_values = model(state_t).squeeze(0).numpy()  # (81,)
        
    #     # Mask invalid moves
    #     q_values_masked = np.full_like(q_values, -1e9)
    #     for r, c in valid_moves:
    #         action_idx = r * N + c  # Updated for 9x9
    #         q_values_masked[action_idx] = q_values[action_idx]
        
    #     best_action_idx = np.argmax(q_values_masked)
    #     return best_action_idx // N, best_action_idx % N  # Updated for 9x9
    

    # def select_max_q_action(self, model, state, valid_moves, N=9):
    #     """Select the action with the highest Q-value for a 9x9 board."""
    #     state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 3, 9, 9)
    #     with torch.no_grad():
    #         q_values = model(state_t).squeeze(0).numpy()  # (81,)

    #     # mask invalid moves
    #     mask = np.full(q_values.shape, -1e9, dtype=np.float32)
    #     for r, c in valid_moves:
    #         action_idx = r * N + c
    #         mask[action_idx] = q_values[action_idx]

    #     best_action_idx = mask.argmax()  # select best move
    #     return best_action_idx // N, best_action_idx % N # convert index to actual move
    
    def select_max_q_action(self, model, state, valid_moves, N=9):
        """Select the action with the highest Q-value."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 3, 9, 9)
        with torch.no_grad():
            # Output van TorchScript-model is hetzelfde formaat, (batch_size, 81)
            q_values = model(state_t).squeeze(0).numpy()  # (81,)

        # Mask invalid moves
        mask = np.full(q_values.shape, -1e9, dtype=np.float32)
        for r, c in valid_moves:
            action_idx = r * N + c
            mask[action_idx] = q_values[action_idx]

        best_action_idx = mask.argmax()
        return best_action_idx // N, best_action_idx % N


    # def compute_best_move(self, game_state: GameState) -> None:
    #     """
    #     Compute and propose the best move for the current game state using iterative deepening and minimax algorithm.

    #     Parameters:
    #         game_state (GameState): The current state of the game.

    #     Returns:
    #         None: Proposes the best move directly using self.propose_move.
    #     """

    #     # Initialize the root node for the game state
    #     root_node = NodeGameState(game_state)
    #     root_node.my_player = root_node.current_player

    #     # Reset the nodes explored counter for this computation
    #     self.nodes_explored = 0

    #     # Retrieve all possible moves for the current state
    #     all_moves = self.get_all_moves(root_node)

    #     if (game_state.board.n, game_state.board.m) == (3,3): 
    #         print('loading dqn model for 9x9 board')
    #         model = self.load_model(filename='team35_9x9_dqn_model.pkl')
    #         squares = [move.square for move in all_moves]
    #         current_board = self.preprocess_board(game_state.board)
    #         state = self.make_state(current_board, player=root_node.my_player)
    #         action = self.select_max_q_action(model, state, squares)
    #         self.propose_move(Move(action, root_node.solved_board_dict[action]))
    #         return
        
    #     elif (game_state.board.n, game_state.board.m) == (3, 2):
    #         print('loading dqn model for 6x6 board')
    #         model = self.load_model(filename='team35_6x6_dqn_model.pkl', input_shape=(6,6), num_actions=36)
    #         squares = [move.square for move in all_moves]
    #         current_board = self.preprocess_board(game_state.board, shape=(6,6))
    #         state = self.make_state(current_board, player=root_node.my_player)
    #         action = self.select_max_q_action(model, state, squares, N=6)
    #         self.propose_move(Move(action, root_node.solved_board_dict[action]))
    #         return

    #     else:
    #         # Randomly propose a move initially to ensure there's always a fallback
    #         self.propose_move(random.choice(all_moves))



    def compute_best_move(self, game_state: GameState) -> None:
        # Initialiseer de root node
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player

        # Retrieve alle mogelijke moves
        all_moves = self.get_all_moves(root_node)

        if (game_state.board.n, game_state.board.m) == (3,3):
            print('Loading TorchScript model for 9x9 board...')
            # Let op de nieuwe functie en bestandsnaam
            model = self.load_torchscript_model(filename='team35_9x9_dqn_model.pt')
            squares = [move.square for move in all_moves]
            current_board = self.preprocess_board(game_state.board)
            state = self.make_state(current_board, player=root_node.my_player)
            action = self.select_max_q_action(model, state, squares, N=9)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return

        elif (game_state.board.n, game_state.board.m) == (3,2):
            print('Loading TorchScript model for 6x6 board...')
            # Stel dat je ook een 6x6 TorchScript-model hebt
            model = self.load_torchscript_model(filename='team35_6x6_dqn_model.pt')
            squares = [move.square for move in all_moves]
            current_board = self.preprocess_board(game_state.board, shape=(6,6))
            state = self.make_state(current_board, player=root_node.my_player)
            action = self.select_max_q_action(model, state, squares, N=6)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return

        else:
            # Fallback: random move
            self.propose_move(random.choice(all_moves))

