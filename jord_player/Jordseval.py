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


weights_opening = [5, 9, 0.01, 5, 2]  
weights_middle = [5, 8, 0.01, 3, 3]
weights_endgame = [5, 7, 0.01, 1, 4]


# Initialize position table globally
position_table = None

def initialize_position_table(board_size: int):
    global position_table
    position_table = generate_position_table(board_size)


def percentage_filled(board: SudokuBoard) -> float:
    '''Calculates the percentage of squares on the board that is filled'''
    N = board.N * board.N
    filled = sum(1 for square in board.squares if square != SudokuBoard.empty)
    return filled / N


def interpolate_weights(percentage_filled: float) -> list[float]:
    """Interpoletes the weights based on the percentage of squares filled"""
    if percentage_filled < 0.33:
        return weights_opening
    elif percentage_filled < 0.66:
        return weights_middle
        # alpha = (percentage_filled - 0.33) / (0.66 - 0.33)
        # return [(1 - alpha) * w_open + alpha * w_middle for w_open, w_middle in zip(weights_opening, weights_middle)]
    else:
        return weights_endgame
        # alpha = (percentage_filled - 0.66) / (1.0 - 0.66)
        # return [(1 - alpha) * w_middle + alpha * w_endgame for w_middle, w_endgame in zip(weights_middle, weights_endgame)]


def evaluate_board(node):
    initialize_position_table(node.board.N)

    pct_filled = percentage_filled(node.board)
    weights = interpolate_weights(pct_filled)

    score_differential = calculate_score_differential(node)
    mobility = normalized_action_potential(node, player=node.my_player)
    region_influence = calculate_region_influence(node, player=node.my_player)
    centrality = calculate_centrality(node)
    minus_one_zones = count_n_minus_1_zones(node, player=node.my_player)
    # zone_control = assess_zone_control(node, player=node.my_player)
    # corner_advantage = evaluate_corner_advantage(node, player=node.my_player)
    # wall_strength = assess_wall_strength(node, player=node.my_player)
    # print(weights[0] * score_differential ,
    #         weights[1] * mobility ,
    #         weights[2] * region_influence ,
    #         weights[3] * centrality,
    #         weights[4] * minus_one_zones)

    return (weights[0] * score_differential +
            weights[1] * mobility +
            weights[2] * region_influence +
            weights[3] * centrality +
            weights[4] * minus_one_zones)# +
            # 15 * zone_control +
            # 10 * corner_advantage +
            # 5 * wall_strength



def calculate_score_differential(node):
    '''Calculates the current score differential, so our points - points of opponent'''

    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]


def calculate_mobility(node):
    ''' Takes as input the game state and looks how many moves our agent can play. If it is not
    their move we make a null-move, i.e. we copy the GameState and change the curren_player, so we
    can check how many adjacient squares we obtain to make a move in the future. '''

    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares())
    
    else: # if it is not our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares())
    

def calculate_opponent_mobility(node):
    ''' Takes as input the game state and looks how many moves the opponent agent can play. If it is not
    our move we make a null-move, i.e. we copy the GameState and change the curren_player, so we
    can check how many adjacient squares we obtain to make a move in the future. '''

    if node.my_player == node.current_player: # if it is our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares())
    
    else: # if it is not our turn
        return len(node.player_squares())
    

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


def count_n_minus_1_zones(game_state: GameState, player: int) -> int:
    """
    Counts n-1 zones that give an advantage to the opponent if it's their turn,
    or to the player if it's the player's turn.
    
    Args:
    - game_state: The current GameState.
    - player: The current player (1 or 2).
    
    Returns:
    - A positive count of advantageous n-1 zones for the current player,
      or a negative count of n-1 zones that favor the opponent.
    """
    N = game_state.board.N  # Board size
    m, n = game_state.board.m, game_state.board.n  # Block dimensions
    empty_squares = {(i, j) for i in range(N) for j in range(N) if game_state.board.get((i, j)) == 0}
    player_occupied = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2
    opponent_occupied = game_state.occupied_squares2 if player == 1 else game_state.occupied_squares1
    current_turn = game_state.current_player

    def count_empty_and_player(squares):
        empty_count = sum(1 for square in squares if square in empty_squares)
        player_count = sum(1 for square in squares if square in player_occupied)
        opponent_count = sum(1 for square in squares if square in opponent_occupied)
        return empty_count, player_count, opponent_count

    n_minus_1_count = 0

    # Check rows
    for i in range(N):
        row = [(i, j) for j in range(N)]
        empty_count, player_count, opponent_count = count_empty_and_player(row)
        if empty_count == 1:
            if current_turn == player and player_count > 0:  # Favorable n-1 for the player
                n_minus_1_count += 1
            elif current_turn != player and opponent_count > 0:  # Dangerous n-1 for the player
                n_minus_1_count -= 1

    # Check columns
    for j in range(N):
        col = [(i, j) for i in range(N)]
        empty_count, player_count, opponent_count = count_empty_and_player(col)
        if empty_count == 1:
            if current_turn == player and player_count > 0:
                n_minus_1_count += 1
            elif current_turn != player and opponent_count > 0:
                n_minus_1_count -= 1

    # Check blocks
    for block_row in range(m):
        for block_col in range(n):
            block = [
                (r, c)
                for r in range(block_row * m, (block_row + 1) * m)
                for c in range(block_col * n, (block_col + 1) * n)
            ]
            empty_count, player_count, opponent_count = count_empty_and_player(block)
            if empty_count == 1:
                if current_turn == player and player_count > 0:
                    n_minus_1_count += 1
                elif current_turn != player and opponent_count > 0:
                    n_minus_1_count -= 1

    return n_minus_1_count


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


def assess_zone_control(game_state: GameState, player: int) -> int:
    """Bereken het aantal regio's (blokken) waar de speler minstens één vierkant bezet, 
    buiten een strategisch gedefinieerde zone."""
    N = game_state.board.N
    m, n = game_state.board.m, game_state.board.n
    player_squares = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2

    # Bepaal de strategische zone (bijv. startzone van de speler)
    start_zone = {0} if player == 1 else {N - 1}
    extended_zone = start_zone.union({min(N - 1, z + 1) for z in start_zone}, {max(0, z - 1) for z in start_zone})

    def block_squares(block_row, block_col):
        return [(r, c) for r in range(block_row * m, (block_row + 1) * m) for c in range(block_col * n, (block_col + 1) * n)]

    control_score = 0
    for block_row in range(m):
        for block_col in range(n):
            if block_row in extended_zone:
                continue  # Vermijd blokken dicht bij de startzone
            if any(square in player_squares for square in block_squares(block_row, block_col)):
                control_score += 1

    return control_score


def evaluate_corner_advantage(game_state: GameState, player: int) -> int:
    """Bereken de controle over strategische hoeken en straf slechte controle."""
    N = game_state.board.N
    player_squares = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2
    opponent_squares = game_state.occupied_squares2 if player == 1 else game_state.occupied_squares1

    critical_corners = [(0, 0), (0, 1), (1, 0)] if player == 1 else [(N - 1, 0), (N - 1, 1), (N - 2, 0)]
    penalty_corner = (0, 0) if player == 1 else (N - 1, 0)

    score = sum(10 for corner in critical_corners if corner in player_squares)
    if penalty_corner in player_squares:
        score -= 100
    if any(corner in opponent_squares for corner in critical_corners):
        return 0

    return score


def normalized_action_potential(game_state: GameState, player: int) -> float:
    """Genormaliseerde mobiliteitsscore."""
    N = game_state.board.N
    if player == game_state.current_player:
        return len(game_state.player_squares()) / N
    else:
        simulated_state = copy.deepcopy(game_state)
        simulated_state.current_player = 3 - game_state.current_player
        return len(simulated_state.player_squares()) / N


def assess_wall_strength(game_state: GameState, player: int) -> int:
    """Bereken bonuspunten voor strategisch sterke zetten langs wanden en rijen."""
    player_squares = game_state.occupied_squares1 if player == 1 else game_state.occupied_squares2
    N = game_state.board.N

    def has_supporting_neighbors(row, col):
        neighbors = [(row + dr, col + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return any(neighbor in player_squares for neighbor in neighbors)

    wall_score = 0
    for row, col in player_squares:
        if col == 0 or col == N - 1:  # Wanden
            if has_supporting_neighbors(row, col):
                wall_score += 2
        if row > 0 and row < N - 1:  # Geen eerste/laatste rij
            if has_supporting_neighbors(row, col):
                wall_score += 1

    return wall_score




