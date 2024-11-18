#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import copy
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    # Example evaluation function
    def evaluate(self, game_state):
        # Replace with your domain-specific evaluation logic
        return game_state.scores[0] - game_state.scores[1]


    def update_score(self, current_score, game_board, n, m, move, player):
        # Convert the string representation of the board into a 2D list

        x, y = move
        game_board[x][y] = player  # Update the board with the player's move

        # Function to check if all elements in a region are integers
        # def is_region_complete(cells):
        #     return all(isinstance(cell, int) for cell in cells)
        def is_region_complete(cells):
            return all(cell != '.' for cell in cells)

        # Count regions completed
        regions_completed = 0

        # Check row
        if is_region_complete(game_board[x]):
            regions_completed += 1

        # Check column
        if is_region_complete([game_board[i][y] for i in range(len(game_board))]):
            regions_completed += 1

        # Check sub-matrix
        sub_x_start, sub_y_start = (x // n) * n, (y // m) * m
        sub_matrix = [
            game_board[i][j]
            for i in range(sub_x_start, sub_x_start + n)
            for j in range(sub_y_start, sub_y_start + m)
        ]
        if is_region_complete(sub_matrix):
            regions_completed += 1

        # Scoring rules
        points_scored = [0, 1, 3, 7][regions_completed]

        # Update player's score
        current_score[player - 1] += points_scored
        return current_score


    def get_children(self, game_state):
        N = game_state.board.N
        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def playable_cells(i, j):
            return game_state.board.get((i, j)) == SudokuBoard.empty and (i, j) in game_state.player_squares()
    
        # Step 1: Extract simplified gameboard
        # print(game_state.board)
        game_board = str(game_state.board)[4:]
        game_board = [line.strip().split() for line in game_board.strip().split("\n")]
        game_board = [[int(cell) if cell.isdigit() else cell for cell in row] for row in game_board]
    
        def find_allowed_values_for_playable_cells(matrix, m, n, target_cells):
            testm = matrix
            N = len(matrix)
            possible_values = {}
            all_numbers = set(range(1, N + 1))
    
            for (row, col) in target_cells:
                if matrix[row][col] != '.':
                    possible_values[(row, col)] = f"Cell already filled with {matrix[row][col]}"
                    continue
    
                # Numbers in the same row
                row_numbers = set()
                for num in matrix[row]:
                    if num != '.':
                        row_numbers.add(num)
    
                # Numbers in the same column
                col_numbers = set()
                for r in range(N):
                    try:
                        if matrix[r][col] != '.':
                            col_numbers.add(matrix[r][col])
                    except:
                        print(r, col)
                        # print(matrix)
                        # print(testm)
                        # print(1010101010101010010101010101010101010101010)
    
                # Determine the region
                region_row = (row // m) * m
                region_col = (col // n) * n
                region_numbers = set()
                for r in range(region_row, region_row + m):
                    for c in range(region_col, region_col + n):
                        try:
                            if matrix[r][c] != '.':
                                region_numbers.add(matrix[r][c])
                        except:
                            print(r, c)
                            # print(matrix)
                            # print(testm)
                            # print(1010101010101010010101010101010101010101010)
    
                # Possible numbers are those not in row, column, or region
                used_numbers = row_numbers.union(col_numbers).union(region_numbers)
                possible = all_numbers - used_numbers
                possible_values[(row, col)] = sorted(possible) if possible else "No possible values"
    
            # all_correct_moves = [Move(key, value) for key in possible_values for value in possible_values[key] if not TabooMove(key, value) in game_state.taboo_moves]
    
            return possible_values
    
        all_playable_cells = [(i, j) for i in range(N) for j in range(N) if playable_cells(i, j)]
        dict_allowed_value_per_cell = find_allowed_values_for_playable_cells(game_board, game_state.board.region_height(), game_state.board.region_width(), all_playable_cells)
        all_moves = [Move(key, value) for key in dict_allowed_value_per_cell for value in dict_allowed_value_per_cell[key] if not TabooMove(key, value) in game_state.taboo_moves]
    
    
    
        all_game_states = []
        rows, cols = game_state.board.region_width(), game_state.board.region_height()
    
        for move in all_moves:
            new_game_state = copy.deepcopy(game_state)
            new_game_state.board.put(move.square, move.value)
            new_game_state.moves.append(move)
            new_game_state.scores = self.update_score(
                new_game_state.scores,
                game_board,  # Strip formatting
                rows,
                cols,
                move.square,
                new_game_state.current_player,
            )
            # Alternate current player and record moves
            if new_game_state.current_player == 1:
                new_game_state.occupied_squares1.append(move.square)
                new_game_state.current_player = 2
            else:
                new_game_state.occupied_squares2.append(move.square)
                new_game_state.current_player = 1
    
            all_game_states.append(new_game_state)
    
        return all_game_states


    def is_terminal(self, game_state):
        N = game_state.board.N
        # A node is terminal if no children (valid moves) exist
        return game_state.occupied_squares1 + game_state.occupied_squares2 == N*N


    def minimax(self, game_state, depth, is_maximizing_player, alpha, beta):
        # Base case: If maximum depth is reached or game state is terminal
        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate(game_state)

        if is_maximizing_player:
            max_eval = float('-inf')
            for child in self.get_children(game_state):
                eval_score = self.minimax(child, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return max_eval
        else:
            min_eval = float('inf')
            for child in self.get_children(game_state):
                eval_score = self.minimax(child, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return min_eval

    def compute_best_move(self, game_state):
        best_move = None
        best_value = float('-inf')

        for depth in range(1, 1000):  # Depth-limited iterative deepening
            #print('depth', depth)
            for child in self.get_children(game_state):
                move_value = self.minimax(child, depth, True, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = child.moves[-1]
                    self.propose_move(best_move)

