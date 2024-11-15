#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

class NodeGameState(GameState):

    def __init__(self, game_state, root_move=None, last_move=None, my_player=None):
        self.__dict__ = game_state.__dict__.copy()
        self.game_board = self.make_matrix(game_state)
        self.root_move = root_move
        self.last_move = last_move
        self.my_player = my_player

    def make_matrix(self, game_state):
        game_board = str(game_state.board)[4:]
        game_board = [line.strip().split() for line in game_board.strip().split("\n")]
        game_board = [[int(cell) if cell.isdigit() else 0 for cell in row] for row in game_board]
        return game_board

    ####### MISS NOG ANDERE FUNCTIE MAKEN OM MATRIX TE BOUWEN ZONDER DE PRINT STATEMENT ############
    # def make_matrix2(self, game_state):
    #     valid_moves = set(game_state.moves) - set(game_state.taboo_moves)
    #     game_board = [[0 for _ in range(game_state.board.N)] for _ in range(game_state.board.N)]
    #     for val_move in valid_moves:
    #         x, y = val_move.square
    #         value = val_move.value
    #         game_board[x][y] = value
    #     return game_board


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    # Example evaluation function
    def evaluate(self, node):
        # Replace with your domain-specific evaluation logic
        return node.scores[node.my_player - 1] - node.scores[node.my_player - 1]


    def update_score(self, node, move):
        # Convert the string representation of the board into a 2D list
        m = node.board.m
        n = node.board.n
        row, col = move.square
        value_move = move.value
        game_board = node.game_board
        current_player = node.current_player
        scores = node.scores
        game_board[row][col] = value_move  # Update the board with the player's move

        # Function to check if all elements in a region are integers
        # def is_region_complete(cells):
        #     return all(isinstance(cell, int) for cell in cells)
        def is_region_complete(cells):
            return all(cell != 0 for cell in cells)

        # Count regions completed
        regions_completed = 0

        # Check row
        if is_region_complete(game_board[row]):
            regions_completed += 1

        # Check column
        if is_region_complete([game_board[i][col] for i in range(len(game_board))]):
            regions_completed += 1


        region_row = (row // m) * m
        region_col = (col // n) * n
        region_matrix = []
        for r in range(region_row, region_row + m):
            for c in range(region_col, region_col + n):
                region_matrix.append(game_board[r][c])

        if is_region_complete(region_matrix):
            regions_completed += 1

        # Scoring rules
        points_scored = [0, 1, 3, 7][regions_completed]

        # Update player's score
        scores[current_player - 1] += points_scored
        return scores

    # def find_allowed_values_for_playable_cells(self, node, target_cells):
    #     matrix = node.game_board
    #     m = node.board.m
    #     n = node.board.n
    #     N = len(matrix)
    #     possible_values = {}
    #     all_numbers = set(range(1, N + 1))
    #
    #     for (row, col) in target_cells:
    #         if matrix[row][col] != 0:
    #             possible_values[(row, col)] = f"Cell already filled with {matrix[row][col]}"
    #             continue
    #
    #         # Numbers in the same row
    #         row_numbers = set()
    #         for num in matrix[row]:
    #             if num != 0:
    #                 row_numbers.add(num)
    #
    #         # Numbers in the same column
    #         col_numbers = set()
    #         for r in range(N):
    #             if matrix[r][col] != 0:
    #                 col_numbers.add(matrix[r][col])
    #
    #
    #         # Determine the region
    #         region_row = (row // m) * m
    #         region_col = (col // n) * n
    #         region_numbers = set()
    #         for r in range(region_row, region_row + m):
    #             for c in range(region_col, region_col + n):
    #                 if matrix[r][c] != 0:
    #                     region_numbers.add(matrix[r][c])
    #
    #
    #         # Possible numbers are those not in row, column, or region
    #         used_numbers = row_numbers.union(col_numbers).union(region_numbers)
    #         possible = all_numbers - used_numbers
    #         possible_values[(row, col)] = sorted(possible) if possible else "No possible values"
    #
    #     return possible_values
    def find_allowed_values(self, node, target_cells):
        matrix = node.game_board
        m = node.board.m
        n = node.board.n
        empty_cell = 0
        N = len(matrix)
        possible_values = {}
        all_numbers = set(range(1, N + 1))

        for (row, col) in target_cells:
            if matrix[row][col] != empty_cell:
                continue

            # Numbers present in the same row
            row_numbers = set(cell for cell in matrix[row] if cell != empty_cell)

            # Numbers present in the same column
            col_numbers = set(matrix[r][col] for r in range(N) if matrix[r][col] != empty_cell)

            # Determine the region's starting indices
            region_row_start = (row // m) * m
            region_col_start = (col // n) * n

            # Numbers present in the same region
            region_numbers = set()
            for r in range(region_row_start, region_row_start + m):
                for c in range(region_col_start, region_col_start + n):
                    if matrix[r][c] != empty_cell:
                        region_numbers.add(matrix[r][c])

            # Possible numbers are those not present in row, column, or region
            used_numbers = row_numbers.union(col_numbers).union(region_numbers)
            possible = set(all_numbers) - set(used_numbers)
            if len(possible)>0:
                possible_values[(row, col)] = sorted(possible)
        #print(possible_values)
        return possible_values


    def get_children(self, node):
        N = node.board.N
        m = node.board.m
        n = node.board.n
        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def playable_cells(i, j):
            return node.board.get((i, j)) == SudokuBoard.empty and (i, j) in node.player_squares()

        all_playable_cells = [(i, j) for i in range(N) for j in range(N) if playable_cells(i, j)]
        dict_allowed_value_per_cell = self.find_allowed_values(node, all_playable_cells)
        all_moves = [Move(key, value) for key in dict_allowed_value_per_cell for value in dict_allowed_value_per_cell[key] if not TabooMove(key, value) in node.taboo_moves]

        all_game_states = []
        for move in all_moves:
            if node.root_move is None:
                new_node = copy.deepcopy(node)
            else:
                new_node = copy.deepcopy(node)
                #new_node = node
            new_node.last_move = move
            new_node.board.put(move.square, move.value)
            new_node.moves.append(move)
            new_node.scores = self.update_score(node, move)
            # Alternate current player and record moves
            if new_node.current_player == 1:
                new_node.occupied_squares1.append(move.square)
                new_node.current_player = 2
            else:
                new_node.occupied_squares2.append(move.square)
                new_node.current_player = 1

            all_game_states.append(new_node)

        return all_game_states

    def is_terminal(self, node):
        N = node.board.N
        # A node is terminal if no children (valid moves) exist
        return len(node.occupied_squares1) + len(node.occupied_squares2) == N*N


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

    def compute_best_move(self, game_state):
        best_move = None
        best_value = float('-inf')
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player
        for depth in range(1, 10):  # Depth-limited iterative deepening
            children = self.get_children(root_node)
            for child in children:
                child.root_move = child.last_move
                move_value = self.minimax(child, depth, False, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = child.root_move
                    self.propose_move(best_move)
