#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .Jordseval import evaluate_board

class NodeGameState(GameState):
    def __init__(self, game_state, root_move=None, last_move=None, my_player=None):
        self.__dict__ = game_state.__dict__.copy()
        self.root_move = root_move
        self.last_move = last_move
        self.my_player = my_player


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    def __init__(self):
        super().__init__()

    def evaluate(self, node):
        # Replace with your domain-specific evaluation logic
        return evaluate_board(node)#return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]


    def is_valid_move_possible(self, node):
        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions

        def is_value_valid(row, col, value):

            # Precompute block starting indices
            block_start_row = (row // m) * m
            block_start_col = (col // n) * n

            for i in range(N):
                # Check row
                if board.get((row, i)) == value:
                    return False

                # Check column
                if board.get((i, col)) == value:
                    return False

                # Check block
                block_row = block_start_row + i // n
                block_col = block_start_col + i % n
                if board.get((block_row, block_col)) == value:
                    return False

            return True

        # Iterate over all cells on the board
        for row in range(N):
            for col in range(N):
                if board.get((row, col)) == 0:  # Empty cell
                    # Check if any value (1 to N) can be placed in this cell
                    for value in range(1, N + 1):
                        if is_value_valid(row, col, value):
                            return True  # Found at least one valid move

        return False  # No valid moves found
    def is_terminal(self, node):
        return not self.is_valid_move_possible(node)

    def respects_rule_C0(self, node, row, col, value):
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

    def playable_square(self, node, i, j, value):
        return node.board.get((i, j)) == SudokuBoard.empty \
            and not TabooMove((i, j), value) in node.taboo_moves \
            and (i, j) in node.player_squares()


    def get_all_moves(self, node):
        N = node.board.N
        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if self.playable_square(node, i, j, value) and self.respects_rule_C0(node, i, j, value)]
        return all_moves


    def calculate_move_score(self, node, move):

        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions
        row, col = move.square  # Get the row and column of the move

        # Precompute block starting indices
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n

        def is_region_complete(values):

            return len(values) == N and len(set(values)) == N and all(value != 0 for value in values)

        # Count completed regions
        completed_regions = 0

        # Check row, column, and block in a single loop
        row_values = []
        col_values = []
        block_values = []

        for i in range(N):
            # Collect row values
            row_values.append(board.get((row, i)))

            # Collect column values
            col_values.append(board.get((i, col)))

            # Collect block values
            block_row = block_start_row + i // n
            block_col = block_start_col + i % n
            block_values.append(board.get((block_row, block_col)))

        # Evaluate completeness of regions
        if is_region_complete(row_values):
            completed_regions += 1
        if is_region_complete(col_values):
            completed_regions += 1
        if is_region_complete(block_values):
            completed_regions += 1

        # Return score based on completed regions
        if completed_regions == 1:
            return 1  # 1 point for 1 region
        elif completed_regions == 2:
            return 3  # 3 points for 2 regions
        elif completed_regions == 3:
            return 7  # 7 points for 3 regions
        else:
            return 0  # No regions completed


    def get_children(self, node):
        all_moves = self.get_all_moves(node)
        all_game_states = []

        if len(all_moves) == 0:
            new_node = copy.deepcopy(node)
            if new_node.current_player == 1:
                new_node.current_player = 2
            else:
                new_node.current_player = 1
            all_game_states.append(new_node)
            return all_game_states

        for move in all_moves:
            new_node = copy.deepcopy(node)
            new_node.last_move = move
            new_node.board.put(move.square, move.value)
            new_node.moves.append(move)
            new_node.scores[new_node.current_player - 1] += self.calculate_move_score(new_node, move)
            if new_node.current_player == 1:
                new_node.occupied_squares1.append(move.square)
                new_node.current_player = 2
            else:
                new_node.occupied_squares2.append(move.square)
                new_node.current_player = 1
            all_game_states.append(new_node)
        return all_game_states


    def minimax(self, node, depth, is_maximizing_player, alpha, beta):
        # Base case: If maximum depth is reached or game state is terminal
        if depth == 0 or self.is_terminal(node):
            return self.evaluate(node)

        if is_maximizing_player:
            max_eval = float('-inf')
            for child in self.get_children(node):
                eval_score = self.minimax(child, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return max_eval
        else:
            min_eval = float('inf')
            for child in self.get_children(node):
                eval_score = self.minimax(child, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return min_eval

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        best_move = None
        best_value = float('-inf')
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player
        children = self.get_children(root_node)
        for child in children:
            child.root_move = child.last_move
            move_value = self.evaluate(child)
            if move_value > best_value:
                best_value = move_value
                best_move = child.root_move
                self.propose_move(best_move)
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        for depth in range(1, 10):
            for child in children:
                move_value = self.minimax(child, depth, False, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = child.root_move
                    self.propose_move(best_move)
            print('depth', depth, 'done','#############################################')







