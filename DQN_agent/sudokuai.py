#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
import pickle
import torch
import os
import sys

# Add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from .dqn import CNNQNetwork
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .evaluation_functions import evaluate_node, calculate_mobility, calculate_score_difference
from .sudoku_solver import SudokuSolver
from .hard_coded_moves import get_heuristic_moves

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
        self.transposition_table = {}
        self.killer_moves = {i: [] for i in range(10)}  # Dictionary to store killer moves for each depth
        self.nodes_explored = 0  # Add this line to initialize the counter


    def evaluate(self, node):
        """
        Evaluate the heuristic value of a node.

        Parameters:
        - node (NodeGameState): The game state node to evaluate.

        Returns:
        - float: The heuristic value of the node.
        """
        return evaluate_node(node)

    def is_terminal(self, node):
        """
        Determine if the game has reached a terminal state (no valid moves left).

        Parameters:
        - node (NodeGameState): The current game state.

        Returns:
        - bool: True if the game is over, False otherwise.
        """
        N=node.board.N
        return len(node.occupied_squares1)+len(node.occupied_squares2)==N*N #not self.is_valid_move_possible(node)


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

    def calculate_move_score(self, node, move):
        """
        Calculate the score obtained by making a particular move.

        Parameters:
        - node (NodeGameState): The current game state.
        - move (Move): The move to evaluate.

        Returns:
        - int: The score obtained from the move.
        """
        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions
        row, col = move.square  # Get the row and column of the move

        # Calculate the starting indices of the block
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n

        def is_region_complete(values):
            """
            Check if a region (row, column, or block) is complete.

            Parameters:
            - values (list of int): The values in the region.

            Returns:
            - bool: True if the region is complete, False otherwise.
            """
            return (len(values) == N
                    and len(set(values)) == N
                    and all(value != SudokuBoard.empty for value in values))

        # Collect values in the row, column, and block
        row_values = [board.get((row, i)) for i in range(N)]
        col_values = [board.get((i, col)) for i in range(N)]
        block_values = [
            board.get((block_start_row + (i // n), block_start_col + (i % n)))
            for i in range(N)
        ]

        # Count completed regions
        completed_regions = sum([
            is_region_complete(row_values),
            is_region_complete(col_values),
            is_region_complete(block_values)
        ])

        # Return score based on completed regions
        if completed_regions == 1:
            return 1  # 1 point for completing 1 region
        elif completed_regions == 2:
            return 3  # 3 points for completing 2 regions
        elif completed_regions == 3:
            return 7  # 7 points for completing all 3 regions
        else:
            return 0  # No regions completed




    def apply_move(self, node, move):
        """
        Apply a move to a node and return the resulting child node.

        Parameters:
        - node (NodeGameState): The current game state.
        - move (Move): The move to apply.

        Returns:
        - NodeGameState: The new game state after applying the move.
        """
        # Create a deep copy of the node to generate a new state
        new_node = copy.deepcopy(node)

        # Update the new node with the applied move
        new_node.last_move = move
        new_node.board.put(move.square, move.value)
        new_node.moves.append(move)

        # Calculate the score resulting from the move and update the player's score
        score = self.calculate_move_score(new_node, move)
        new_node.scores[new_node.current_player - 1] += score

        # Update occupied squares and switch the current player
        if new_node.current_player == 1:
            new_node.occupied_squares1.append(move.square)
            new_node.current_player = 2
        else:
            new_node.occupied_squares2.append(move.square)
            new_node.current_player = 1

        # Return the new node representing the updated game state
        return new_node


    def minimax(self, node, depth, is_maximizing_player, alpha, beta):
        """
        Perform the Minimax algorithm with Alpha-Beta pruning, transposition table, and killer moves optimization.

        Parameters:
        - node (NodeGameState): The current game state.
        - depth (int): The depth limit for the search.
        - is_maximizing_player (bool): True if it's the maximizing player's turn.
        - alpha (float): Alpha value for pruning (best already explored option along the path to the root for the maximizer).
        - beta (float): Beta value for pruning (best already explored option along the path to the root for the minimizer).

        Returns:
        - float: The evaluated score of the node.
        """
        # Increment the counter for nodes explored
        self.nodes_explored += 1

        # Generate a unique key for the node to check in the transposition table
        node_key = (node.hash_key(), depth, is_maximizing_player)

        # Check if the current node evaluation is already cached in the transposition table
        if node_key in self.transposition_table:
            return self.transposition_table[node_key]

        # Base case: If maximum depth is reached or the node is a terminal state, evaluate the node
        if depth == 0 or self.is_terminal(node):
            score = self.evaluate(node)
            self.transposition_table[node_key] = score  # Cache the result
            return score

        # Retrieve all moves and prioritize killer moves
        moves = self.get_all_moves(node)
        prioritized_moves = self.killer_moves[depth] + [move for move in moves if move not in self.killer_moves[depth]]

        if is_maximizing_player:
            # Maximizing player's turn
            max_eval = float('-inf')

            for move in prioritized_moves:
                # Apply the move to get the child node
                child = self.apply_move(node, move)

                # Recursively call minimax for the child node
                eval_score = self.minimax(child, depth - 1, False, alpha, beta)

                # Update the maximum evaluation score and alpha value
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # Perform alpha-beta pruning if applicable
                if beta <= alpha:
                    # Add the move to the killer moves list if not already present
                    if move not in self.killer_moves[depth]:
                        if len(self.killer_moves[depth]) >= 2:  # Maintain at most two killer moves
                            self.killer_moves[depth].pop(0)
                        self.killer_moves[depth].append(move)
                    break

            # Cache the result and return the maximum evaluation score
            self.transposition_table[node_key] = max_eval
            return max_eval

        else:
            # Minimizing player's turn
            min_eval = float('inf')

            for move in prioritized_moves:
                # Apply the move to get the child node
                child = self.apply_move(node, move)

                # Recursively call minimax for the child node
                eval_score = self.minimax(child, depth - 1, True, alpha, beta)

                # Update the minimum evaluation score and beta value
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                # Perform alpha-beta pruning if applicable
                if beta <= alpha:
                    # Add the move to the killer moves list if not already present
                    if move not in self.killer_moves[depth]:
                        if len(self.killer_moves[depth]) >= 2:  # Maintain at most two killer moves
                            self.killer_moves[depth].pop(0)
                        self.killer_moves[depth].append(move)
                    break

            # Cache the result and return the minimum evaluation score
            self.transposition_table[node_key] = min_eval
            return min_eval


    def load_model(self, filename="dqn_model_2x2.pkl"):
        """Load the trained model from the 'models' folder in the same directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Map van dit script
        models_dir = os.path.join(script_dir, "models")  # Pad naar 'models' map
        file_path = os.path.join(models_dir, filename)  # Volledige pad naar modelbestand

        # Model laden
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Pas de input_shape en num_actions aan overeenkomstig je nieuwe architectuur
        model = CNNQNetwork(input_shape=(9, 9), num_actions=81)
        model.load_state_dict(data["policy_state_dict"])
        model.eval()
        print(f"Model loaded from {file_path}")
        return model
    

    def preprocess_board(self, original_board):
        # Convert squares to a 9x9 numpy array
        board_array = np.array(original_board.squares).reshape((9, 9))

        # Mark squares owned by player 1 as 1
        board_array[board_array > 0] = 1

        # Mark squares owned by player 2 as -1
        board_array[board_array < 0] = -1

        # Zeros remain 0 (empty)
        return board_array


    def make_state(self, board, player=1):
        # Here board is now shape (9, 9)
        p1_channel = (board == player).astype(np.float32)   # shape (9,9)
        p2_channel = (board == -player).astype(np.float32)  # shape (9,9)
        empty_channel = (board == 0).astype(np.float32)     # shape (9,9)
        
        # Now stack -> shape (3, 9, 9)
        state = np.stack([p1_channel, p2_channel, empty_channel], axis=0)
        return state


    def select_max_q_action(self, model, state, valid_moves):
        """Select the action with the highest Q-value for a 9x9 board."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (1, 3, 9, 9)
        with torch.no_grad():
            q_values = model(state_t).squeeze(0).numpy()  # (81,)
        
        # Mask invalid moves
        q_values_masked = np.full_like(q_values, -1e9)
        for r, c in valid_moves:
            action_idx = r * 9 + c  # Updated for 9x9
            q_values_masked[action_idx] = q_values[action_idx]
        
        best_action_idx = np.argmax(q_values_masked)
        return best_action_idx // 9, best_action_idx % 9  # Updated for 9x9


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

        print(game_state.board.n, game_state.board.m)

        if (game_state.board.n, game_state.board.m) == (3,3) and sum(1 for square in game_state.board.squares if square != 0) <= 0.6 * len(game_state.board.squares) : # maybe add (and total moves < ...)
            print('loading dqn model for 9x9 board')
            model = self.load_model(filename='team35_dqn_model.pkl')
            squares = [move.square for move in all_moves]
            current_board = self.preprocess_board(copy.deepcopy(game_state.board))
            state = self.make_state(current_board, player=root_node.my_player)
            action = self.select_max_q_action(model, state, squares)
            self.propose_move(Move(action, root_node.solved_board_dict[action]))
            return
        
    
        elif ((game_state.board.n, game_state.board.m) == (2, 3) or (game_state.board.n, game_state.board.m) == (2, 2)) and (root_node.my_player == 2): #2,2 werkt nog niet
            all_moves = self.get_all_moves(root_node)
            last_move = root_node.last_move

            if last_move is not None:
                (r_p1, c_p1) = last_move.square
                val_p1 = last_move.value  # P1's digit
                N = root_node.board.N  # Total board size
                print(N)

                # Horizontale spiegeling
                horiz_r = (N - 1) - r_p1
                horiz_c = c_p1
                horiz_sq = (horiz_r, horiz_c)

                if horiz_sq in [move.square for move in all_moves]:
                    # Probeer een waarde die de tegenstander al eerder heeft gespeeld
                    opponent_values = {move.value for move in root_node.moves if move.player == 1}

                    # Check of een waarde van de tegenstander gebruikt kan worden
                    for opp_value in opponent_values:
                        test_board = copy.deepcopy(root_node.board)
                        test_board.put(horiz_sq, opp_value)
                        solver = SudokuSolver(test_board.squares, test_board.N, test_board.m, test_board.n)

                        if solver.is_solvable():
                            self.propose_move(Move(horiz_sq, opp_value))
                            return

                    # Als geen waarde van de tegenstander mogelijk is, gebruik een veilige waarde
                    if horiz_sq in root_node.solved_board_dict:
                        safe_value = root_node.solved_board_dict[horiz_sq][0]  # Neem de eerste veilige waarde
                        self.propose_move(Move(horiz_sq, safe_value))
                        return

            # Als geen spiegeling mogelijk is, kies een willekeurige geldige zet
            self.propose_move(random.choice(all_moves))
            return

        else:
        
            self.propose_move(random.choice(all_moves))

            # Randomly propose a move initially to ensure there's always a fallback
            self.propose_move(random.choice(all_moves))

            # Check for heuristic-based moves and propose if available
            heuristic_move = get_heuristic_moves(root_node)
            if heuristic_move:
                self.propose_move(heuristic_move)
                return

            # Initialize the best value to a very low number to track the maximum score
            best_value = float('-inf')

            # Perform iterative deepening up to a specified maximum depth
            for depth in range(10):  # Depth is set to 10 by default
                counter_nodes = 0  # Initialize a counter for nodes explored at the current depth

                for move in all_moves:
                    # Apply the move to generate the child node
                    child = self.apply_move(root_node, move)

                    # Evaluate the child node using the minimax algorithm with alpha-beta pruning
                    move_value = self.minimax(child, depth, False, float('-inf'), float('inf'))

                    # Update the best move if the current move is better
                    if move_value > best_value:
                        best_value = move_value
                        best_move = move

                        # Propose the best move found so far
                        self.propose_move(best_move)

                    # Increment the counter for nodes explored
                    counter_nodes += 1
                    #print(counter_nodes)

                # Output the progress for the current depth
                #print(f'Depth {depth + 1} search complete.')
                #print(f'Nodes explored at depth {depth + 1}: {self.nodes_explored}')

                # Reset nodes explored counter for tracking nodes at the next depth level
                self.nodes_explored = 0
