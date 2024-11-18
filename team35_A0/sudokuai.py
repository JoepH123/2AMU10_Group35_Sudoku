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
    def is_terminal(self, node):
        N = node.board.N
        # A node is terminal if no children (valid moves) exist
        return len(node.occupied_squares1) + len(node.occupied_squares2) == N*N

    def respects_rule_C0(self, node, row, col, value):
        board = node.board
        N = node.board.N  # Size of the grid (N = n * m)
        n, m = node.board.n, node.board.m  # Block dimensions
        # Check the row
        for c in range(N):
            if board.get((row, c)) == value:  # Assuming `board.get(r, c)` retrieves the value at (r, c)
                return False

        # Check the column
        for r in range(N):
            if board.get((r, col)) == value:
                return False

        # Check the block
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n
        for r in range(block_start_row, block_start_row + m):
            for c in range(block_start_col, block_start_col + n):
                if board.get((r, c)) == value:
                    return False
        return True

    def playable_square(self, game_state, i, j, value):
        return game_state.board.get((i, j)) == SudokuBoard.empty \
            and not TabooMove((i, j), value) in game_state.taboo_moves \
            and (i, j) in game_state.player_squares()

    def get_all_moves(self, game_state):
        N = game_state.board.N
        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if self.playable_square(game_state, i, j, value) and self.respects_rule_C0(game_state, i, j, value)]
        return all_moves


    def calculate_move_score(self, state, move):
        N = state.board.N  # Size of the grid
        n, m = state.board.n, state.board.m  # Block dimensions
        row, col = move.square  # Get the row and column of the move

        def is_region_complete(cells):
            values = [state.board.get((r, c)) for r, c in cells]
            return len(values) == N and len(set(values)) == N  and  all(value != 0 for value in values)

        # Helper to generate cells in a row
        def get_row_cells(row):
            return [(row, c) for c in range(N)]

        # Helper to generate cells in a column
        def get_col_cells(col):
            return [(r, col) for r in range(N)]

        # Helper to generate cells in a block
        def get_block_cells(block_row, block_col):
            return [
                (r, c)
                for r in range(block_row * m, (block_row + 1) * m)
                for c in range(block_col * n, (block_col + 1) * n)
            ]

        # Check if the move completes any regions
        completed_regions = 0
        if is_region_complete(get_row_cells(row)):
            completed_regions += 1
        if is_region_complete(get_col_cells(col)):
            completed_regions += 1
        block_row = row // m
        block_col = col // n
        if is_region_complete(get_block_cells(block_row, block_col)):
            completed_regions += 1

        # Calculate score based on completed regions
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
        for depth in range(1, 10):
            print('depth', depth, '#############################################')
            for child in children:
                move_value = self.minimax(child, depth, False, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = child.root_move
                    self.propose_move(best_move)






