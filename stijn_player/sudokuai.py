#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .evaluate_functions import evaluate_node, calculate_mobility, calculate_score_difference
from .sudoku_solver import SudokuSolver
from .hard_coded_moves import opening_game_phase_moves


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

        This could be based on a serialized representation of the board state,
        current player, and other relevant attributes.
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
        new_node = copy.deepcopy(node)
        new_node.last_move = move
        new_node.board.put(move.square, move.value)
        new_node.moves.append(move)
        # Update the score for the current player
        score = self.calculate_move_score(new_node, move)
        new_node.scores[new_node.current_player - 1] += score
        # Update occupied squares and switch player
        if new_node.current_player == 1:
            new_node.occupied_squares1.append(move.square)
            new_node.current_player = 2
        else:
            new_node.occupied_squares2.append(move.square)
            new_node.current_player = 1
        return new_node

    def minimax(self, node, depth, is_maximizing_player, alpha, beta):
        """
        Perform the Minimax algorithm with Alpha-Beta pruning.

        Parameters:
        - node (NodeGameState): The current game state.
        - depth (int): The depth limit for the search.
        - is_maximizing_player (bool): True if it's the maximizing player's turn.
        - alpha (float): Alpha value for pruning.
        - beta (float): Beta value for pruning.

        Returns:
        - float: The evaluated score of the node.
        """
        self.nodes_explored += 1
        node_key = (node.hash_key(), depth, is_maximizing_player)  # Unique identifier for caching
        if node_key in self.transposition_table:
            return self.transposition_table[node_key]

        # Base case: If maximum depth is reached or game state is terminal
        if depth == 0 or self.is_terminal(node):
            score = self.evaluate(node)
            self.transposition_table[node_key] = score
            return score

        moves = self.get_all_moves(node)
        prioritized_moves = self.killer_moves[depth] + [move for move in moves if move not in self.killer_moves[depth]]

        if is_maximizing_player:
            max_eval = float('-inf')

            for move in prioritized_moves:
                child = self.apply_move(node, move)
                eval_score = self.minimax(child, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if move not in self.killer_moves[depth]:
                        if len(self.killer_moves[depth]) >= 2:  # Keep only two killer moves
                            self.killer_moves[depth].pop(0)
                        self.killer_moves[depth].append(move)
                    break  # Alpha-Beta Pruning
            self.transposition_table[node_key] = max_eval
            return max_eval

        else:
            min_eval = float('inf')

            for move in prioritized_moves:
                child = self.apply_move(node, move)
                eval_score = self.minimax(child, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if move not in self.killer_moves[depth]:
                        if len(self.killer_moves[depth]) >= 2:  # Keep only two killer moves
                            self.killer_moves[depth].pop(0)
                        self.killer_moves[depth].append(move)
                    break  # Alpha-Beta Pruning
            self.transposition_table[node_key] = min_eval
            return min_eval

    def compute_best_move(self, game_state: GameState) -> None:
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player

        self.nodes_explored = 0  # Reset the counter before the search
        all_moves = self.get_all_moves(root_node)
        self.propose_move(random.choice(all_moves))

        opening_move = opening_game_phase_moves(root_node)
        if opening_move:
            self.propose_move(opening_move)
            return

        best_value = float('-inf')
        for depth in range(10):
            counter_nodes=0
            for move in all_moves:
                child = self.apply_move(root_node, move)
                move_value = self.minimax(child, depth, False, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
                    self.propose_move(best_move)
                counter_nodes+=1
                #print(counter_nodes)
            print(f'Depth {depth+1} search complete.')
            print(f'Nodes explored at depth {depth+1}: {self.nodes_explored}')
            self.nodes_explored = 0  # Reset if tracking per depth

