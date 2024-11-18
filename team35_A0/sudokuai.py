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

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def playable_cells(i, j):
            return game_state.board.get((i, j)) == SudokuBoard.empty and (i, j) in game_state.player_squares()
        
        # Step 1: Extract simplified gameboard
        game_board = str(game_state.board)[4:-1]
        game_board = [line.strip().split() for line in game_board.strip().split("\n")]
        game_board = [[int(cell) if cell.isdigit() else cell for cell in row] for row in game_board]
        
        def find_allowed_values_for_playable_cells(matrix, m, n, target_cells):
            """
            matrix : gamestate_board (with size N)
                Example:
                6x6 matrix with some pre-filled numbers
                "." - denotes empty cells
                "+" and "-" only indicate the player that put the move
                matrix = [
                    [1+, '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.','.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', 1-]
                ]
            m : region_height (number of rows in region)
                Example: m=2
            n : region_widths (number of columns in region)
                Example: n=3
            target_cells : playable_cells (cells that player is allowed to put a number in non-taboo, adjacent, in-bound)
                Example: Player "+": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1)]
                Example: Player "-": [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (4, 4), (4, 5)]
            
            Output: 
                Since only 1 is in two completely different regions, all target cells can receive all possible 
            """
            N = len(matrix)
            possible_values = {}
            all_numbers = set(range(1, N + 1))
            print(target_cells)
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
                    if matrix[r][col] != '.':
                        col_numbers.add(matrix[r][col])
                
                # Determine the region
                region_row = (row // m) * m
                region_col = (col // n) * n
                region_numbers = set()
                for r in range(region_row, region_row + m):
                    for c in range(region_col, region_col + n):
                        if matrix[r][c] != '.':
                            region_numbers.add(matrix[r][c])
                
                # Possible numbers are those not in row, column, or region
                used_numbers = row_numbers.union(col_numbers).union(region_numbers)
                possible = all_numbers - used_numbers
                possible_values[(row, col)] = sorted(possible) if possible else "No possible values"

            # all_correct_moves = [Move(key, value) for key in possible_values for value in possible_values[key] if not TabooMove(key, value) in game_state.taboo_moves]
            
            return possible_values

        all_playable_cells = [(i, j) for i in range(N) for j in range(N) if playable_cells(i, j)]
        dict_allowed_value_per_cell = find_allowed_values_for_playable_cells(game_board, game_state.board.region_height(), game_state.board.region_width(), all_playable_cells)
        all_correct_moves = [Move(key, value) for key in dict_allowed_value_per_cell for value in dict_allowed_value_per_cell[key] if not TabooMove(key, value) in game_state.taboo_moves]

        move = random.choice(all_correct_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_correct_moves))


# class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
#     """
#     Sudoku AI that computes a move using minimax with iterative deepening.
#     """

#     def __init__(self):
#         super().__init__()

#     def compute_best_move(self, game_state: GameState) -> None:
#         N = game_state.board.N
#         print(f"Start function; N={N}")

#         # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
#         def possible(i, j, value):
#             return game_state.board.get((i, j)) == SudokuBoard.empty \
#                    and not TabooMove((i, j), value) in game_state.taboo_moves \
#                        and (i, j) in game_state.player_squares()
        
#         def get_all_moves():
#             return [Move((i, j), value) for i in range(N) for j in range(N)
#                      for value in range(1, N+1) if possible(i, j, value)]

#         # def is_valid_move(i, j, value):
#         #     """Checks whether a move is valid."""
#         #     return (game_state.board.get((i, j)) == SudokuBoard.empty
#         #             and not TabooMove((i, j), value) in game_state.taboo_moves
#         #             and (i, j) in game_state.player_squares())

#         # def get_all_moves():
#         #     """Generates all possible moves."""
#         #     return [Move((i, j), value) for i in range(N) for j in range(N)
#         #             for value in range(1, N + 1) if is_valid_move(i, j, value)]

#         def evaluate(state):
#             """Simple evaluation function."""
#             current_player = state.current_player
#             opponent_player = 2 - state.current_player
#             return state.scores[current_player] - state.scores[opponent_player]
        
#         def simulate_move(game_state: GameState, move: Move) -> GameState:
#             """
#             Applies a move to the given game state and returns a new game state reflecting the move.
            
#             @param game_state: The current game state.
#             @param move: The move to apply.
#             @return: A new GameState object with the move applied.
#             """
#             # Create a deep copy of the current game state to avoid mutating the original
#             new_game_state = copy.deepcopy(game_state)
            
#             # Apply the move to the board
#             new_game_state.board.put(move.square, move.value)
            
#             # Append the move to the move history
#             new_game_state.moves.append(move)
            
#             # Update occupied squares if playmode is not classic
#             if not new_game_state.is_classic_game():
#                 if new_game_state.current_player == 1:
#                     if new_game_state.occupied_squares1 is not None:
#                         new_game_state.occupied_squares1.append(move.square)
#                 else:
#                     if new_game_state.occupied_squares2 is not None:
#                         new_game_state.occupied_squares2.append(move.square)
            
#             # Update scores based on the move
#             # (Assuming the move's score is already determined; adjust as needed)
#             # For demonstration, let's assume each valid move adds 1 point
#             new_game_state.scores[new_game_state.current_player - 1] += 1
            
#             # Switch the current player
#             new_game_state.current_player = 3 - new_game_state.current_player
            
#             return new_game_state


#         def minimax(state, depth, is_maximizing):
#             """Recursive minimax function."""
#             if depth == 0:  # or state.is_terminal(): assume function would not be called if state was terminal (if game had ended)
#                 return evaluate(state), None
#             state_copied = copy.deepcopy(state)
#             possible_moves = get_all_moves()
#             if is_maximizing:
#                 max_eval = float('-inf')
#                 best_move = None
#                 for move in possible_moves:
#                     new_state = simulate_move(state_copied, move)
#                     eval_score, _ = minimax(new_state, depth - 1, False)
#                     if eval_score > max_eval:
#                         max_eval = eval_score
#                         best_move = move
#                 return max_eval, best_move
#             else:
#                 min_eval = float('inf')
#                 best_move = None
#                 for move in possible_moves:
#                     new_state = simulate_move(state_copied, move)
#                     eval_score, _ = minimax(new_state, depth - 1, True)
#                     if eval_score < min_eval:
#                         min_eval = eval_score
#                         best_move = move
#                 return min_eval, best_move

#         def iterative_deepening(state, max_depth, time_limit):
#             """Performs iterative deepening search."""
#             start_time = time.time()
#             best_move = None

#             for depth in range(1, max_depth + 1):
#                 # Stop searching if time runs out
#                 if time.time() - start_time > time_limit:
#                     break

#                 _, move = minimax(state, depth, is_maximizing=True)
#                 if move:
#                     best_move = move

#                 # Output progress (optional)
#                 self.propose_move(best_move)
#                 print(f"Depth {depth} completed. Best move so far: {best_move}")

#             return best_move
#         print("start process!!!!!!")
#         # Perform iterative deepening
#         time_limit = 2  # Set a time limit for the search in seconds
#         max_depth = 5   # Maximum search depth
#         iterative_deepening(game_state, max_depth, time_limit)


