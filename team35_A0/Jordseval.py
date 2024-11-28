import random
import time
import copy
import numpy as np

import sys
import os
# Voeg de map toe waar 'competitive_sudoku' staat
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


# Initialize position table globally
position_table = None

def initialize_position_table(board_size: int):
    global position_table
    position_table = generate_position_table(board_size)


def evaluate_board(node):
    initialize_position_table(node.board.N)
    score_differential = calculate_score_differential(node)
    mobility = calculate_mobility(node)
    region_influence = calculate_region_influence(node, player=node.my_player)
    centrality = calculate_centrality(node)
    print(15 * score_differential, 5 * mobility, 0.1 * region_influence, 10 * centrality )
  
    return 15 * score_differential + 5 * mobility + 0.1 * region_influence + 15 * centrality  


def calculate_score_differential(node):
    '''Calculates the current score differential, so our points - points of opponent'''

    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]


def calculate_mobility(node):
    ''' Takes as input the game state and looks how many moves our agent can play. If it is not
    our move we make a null-move, i.e. we copy the GameState and change the curren_player, so we
    can check how many adjacient squares we obtain to make a move in the future. '''

    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares())
    
    else: # if it is not our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares())
    

def calculate_region_influence(game_state: GameState, player: int) -> int:
    '''      Calculates a player's influence in rows, columns, and blocks based on how many squares
    they occupy in each region. Higher influence means the player is closer to completing regions.For
    each region the influence is calculated as the amount of squares squared, so that more squares in one
    region score more than having less squares in more regions.'''

    N = game_state.board.N
    m, n = game_state.board.m, game_state.board.n
    occupied_squares = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2
    influence = 0

    # Helper to calculate influence for a single region
    def region_score(squares):
        player_count = sum(1 for square in squares if square in occupied_squares)
        return player_count ** 2  # Weight heavily for higher occupation

    # Calculate influence for rows
    for i in range(N):
        row = [(i, j) for j in range(N)]
        influence += region_score(row)

    # Calculate influence for columns
    for j in range(N):
        col = [(i, j) for i in range(N)]
        influence += region_score(col)

    # Calculate influence for blocks
    for block_row in range(m):
        for block_col in range(n):
            block = [
                (r, c)
                for r in range(block_row * m, (block_row + 1) * m)
                for c in range(block_col * n, (block_col + 1) * n)
            ]
            influence += region_score(block)

    return influence


def generate_position_table(board_size: int) -> np.ndarray:
    """ Generates a position table for a Sudoku board where central squares
    are given higher values to prioritize central positioning. """

    center = board_size / 2 - 0.5  # Exact center for odd or even sizes
    position_table = np.zeros((board_size, board_size))

    for i in range(board_size):
        for j in range(board_size):
            # Calculate the "distance from the center" and invert it for higher center value
            distance = abs(center - i) + abs(center - j)
            position_table[i, j] = 1 / (1 + distance)  # Higher value for closer to center

    # Normalize values so they range between 0 and 1
    position_table = position_table / position_table.max()
    
    return position_table


def calculate_centrality(node):
    """Calculates the centrality score for the current player based on the position table."""
    global position_table
    if position_table is None:
        raise ValueError("Position table has not been initialized.")

    occupied_squares = node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    centrality_score = sum(position_table[square[0], square[1]] for square in occupied_squares)
    
    return centrality_score






