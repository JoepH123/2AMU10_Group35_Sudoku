import random
import numpy as np
from .MCT_functions import *

# def get_legal_moves_rollout(board, current_player, N):
#     """
#     Returns a list of (x, y) coordinates for all legal moves for current player.
#     A legal move is any empty cell (value = 0) that is in row 0
#     OR adjacent (in any of the 8 directions) to at least one cell with current player.
#     """
#
#     # Mark which cells are neighbors of a cell with a '1'
#     neighbors_of_current_player = np.zeros((N, N), dtype=bool)
#
#     # Directions (dx, dy) to cover 8 neighbors (including diagonals)
#     directions = [(-1, -1), (-1,  0), (-1,  1),
#                   ( 0, -1),           ( 0,  1),
#                   ( 1, -1), ( 1,  0), ( 1,  1)]
#
#     # For each cell that contains 1, mark its neighbors
#     current_player_positions = np.argwhere(board == current_player)  # array of [x,y] where board[x,y] == 1
#     for x, y in current_player_positions:
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < N and 0 <= ny < N:
#                 neighbors_of_current_player[nx, ny] = True
#
#     # Build a boolean mask of empty cells
#     empty_mask = (board == 0)
#
#     # Build a boolean mask for row 0
#     row0_mask = np.zeros((N, N), dtype=bool)
#     if current_player==1:
#         row0_mask[0, :] = True
#     else:
#         row0_mask[N-1, :] = True
#
#     # A legal move must be empty AND (in row 0 OR neighbor_of_1)
#     legal_mask = empty_mask & (row0_mask | neighbors_of_current_player)
#
#     # Extract (x, y) coordinates from the mask
#     legal_moves = list(zip(*np.where(legal_mask)))
#
#     return legal_moves

def get_legal_moves_rollout(board, current_player, N):
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

def calculate_score_rollout(board, move, m, n):
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

def rollout(node):
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
        moves = get_legal_moves_rollout(board, current_player, N)
        if len(moves)>0:
            move = random.choice(moves)
            board[move] = current_player
            scores[current_player-1] += calculate_score_rollout(board, move, m, n)

        if current_player == 1:
            current_player = 2
        else:
            current_player = 1

    winner = get_winner(scores)
    return winner  # +1 (Player1), -1 (Player2), or 0 (draw)



