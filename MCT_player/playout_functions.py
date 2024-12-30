import random
import numpy as np
from .MCT_functions import *

def get_legal_moves_playout(board, current_player, N):
    """
    Returns a list of (x, y) coordinates for all legal moves for the current player.
    A legal move is:
        - any empty cell (value = 0) that is in row 0 if current_player=1, or row N-1 if current_player=2,
        OR
        - adjacent (in any of the 8 directions) to at least one cell with current_player.
    """
    # Directions (dx, dy) to cover 8 neighbors
    directions = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ])

    # Build a boolean mask of empty cells
    empty_mask = (board == 0)

    # Build a boolean mask for the row of interest (row 0 for player1, row N-1 for player2)
    row0_mask = np.zeros((N, N), dtype=bool)
    if current_player == 1:
        row0_mask[0, :] = True
    else:
        row0_mask[N - 1, :] = True

    # Vectorized approach to find neighbors of current_player
    # -------------------------------------------------------
    # 1) Get all positions of current_player
    # 2) Broadcast directions and add
    # 3) Check bounds and mark in a boolean array
    current_positions = np.argwhere(board == current_player)
    if len(current_positions) == 0:
        # No cells are occupied by current_player yet;
        # only row0_mask cells are legal (assuming the game allows the first move there).
        legal_mask = empty_mask & row0_mask
        return list(zip(*np.where(legal_mask)))

    # Expand current_positions for each of the 8 directions:
    # shape: (num_positions, 1, 2) + (1, 8, 2) => (num_positions, 8, 2)
    neighbors = current_positions[:, None, :] + directions[None, :, :]
    # Clip coordinates to remain in-bounds
    valid_neighbors = (neighbors[:, :, 0] >= 0) & (neighbors[:, :, 0] < N) & \
                 (neighbors[:, :, 1] >= 0) & (neighbors[:, :, 1] < N)

    # Flatten the valid neighbors into a list of coordinates
    valid_x = neighbors[:, :, 0][valid_neighbors].ravel()
    valid_y = neighbors[:, :, 1][valid_neighbors].ravel()

    # Create a boolean mask of neighbors of current_player
    neighbors_of_current_player = np.zeros((N, N), dtype=bool)
    neighbors_of_current_player[valid_x, valid_y] = True

    # Combine conditions: empty AND (in row0_mask OR neighbors_of_current_player)
    legal_mask = empty_mask & (row0_mask | neighbors_of_current_player)

    # Extract (x, y) coordinates
    legal_moves = list(zip(*np.where(legal_mask)))
    return legal_moves

def calculate_score_playout(board, move, m, n):
    x, y = move

    row_complete = np.all(board[x, :] != 0)
    col_complete = np.all(board[:, y] != 0)

    region_row = (x // m) * m
    region_col = (y // n) * n

    region = board[region_row:region_row + m, region_col:region_col + n]
    region_complete = np.all(region != 0)

    regions_completed = int(row_complete) + int(col_complete) + int(region_complete)

    if regions_completed == 0:
        return 0
    elif regions_completed == 1:
        return 1
    elif regions_completed == 2:
        return 3
    elif regions_completed == 3:
        return 7
    else:
        return 0

def get_winner(scores):
    score_p1, score_p2 = scores
    if score_p1 > score_p2:
        return +1   # Player 1 wins
    elif score_p2 > score_p1:
        return -1   # Player 2 wins
    else:
        return 0    # Draw

def move_probability_score(moves, board, m, n):
    N = m*n
    move_score_dict = {}
    for move in moves:
        test_board = board.copy()
        test_board[move] = 3
        x, y = move

        row_complete = np.all(test_board[x, :] != 0)
        col_complete = np.all(test_board[:, y] != 0)

        region_row = (x // m) * m
        region_col = (y // n) * n

        region = test_board[region_row:region_row + m, region_col:region_col + n]
        region_complete = np.all(region != 0)

        regions_completed = int(row_complete) + int(col_complete) + int(region_complete)


        if regions_completed == 0:
            move_score_dict[move] = 1
        elif regions_completed == 1:
            move_score_dict[move] = 3
        elif regions_completed == 2:
            move_score_dict[move] = 5
        elif regions_completed == 3:
            move_score_dict[move] = 10
        else:
            move_score_dict[move] = 1

    return move_score_dict

def move_probabilty_mobility(board, moves, N):


    # Directions for adjacent cells (including diagonals)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    result = {}

    for coord in moves:
        x, y = coord
        count = 0

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the new coordinate is within bounds and has a value of 0
            if 0 <= nx < N and 0 <= ny < N:
                if board[nx][ny] == 0:
                    count += 1

        result[coord] = count
    return result

def make_form_move_probability_dicts_list(dicts):
    all_move_frequency_lists = []
    for dict in dicts:
        move_frequency_list = []
        for item, frequency in dict.items():
            move_frequency_list.extend([item] * frequency)
        all_move_frequency_lists.extend(move_frequency_list)
    return all_move_frequency_lists

def playout(node):
    """
    Simulate a random play-out from the given state and return the outcome
    from the perspective of 'state.current_player'.
    """
    board = node.board.copy()
    scores = node.scores.copy()
    current_player = node.current_player
    N = node.N
    m = node.m
    n = node.n
    while not np.all(board != 0):
        moves = get_legal_moves_playout(board, current_player, N)
        if len(moves)>0:
            probability_moves_score_dict = move_probability_score(moves, board, m, n)
            probability_moves_mobility_dict = move_probabilty_mobility(board, moves, N)
            probability_moves = make_form_move_probability_dicts_list([probability_moves_score_dict,
                                                                       probability_moves_mobility_dict])
            move = random.choice(probability_moves)
            board[move] = current_player
            scores[current_player-1] += calculate_score_playout(board, move, m, n)

        if current_player == 1:
            current_player = 2
        else:
            current_player = 1

    winner = get_winner(scores)
    return winner  # +1 (Player1), -1 (Player2), or 0 (draw)


