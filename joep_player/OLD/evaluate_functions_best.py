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

# python simulate_game.py --first=joep_player --second=jord_player
# python simulate_game.py --first=jord_player --second=joep_player
# python simulate_game.py --first=joep_player --second=stijn_player

def evaluate_node(node):
    ''' Combining the evaluation functions '''
    stage, weights = weights_at_game_stage(node)
    score_differential = calculate_score_differential(node)
    score_second_to_last_placement_in_region = evaluate_second_to_last_placement_in_region(node)
    score_last_placement_in_region = evaluate_last_placement_in_region(node)  # also in score differential, might be obsolete
    mobility = positively_evaluate_mobility(node)
    centrality = evaluate_central_control(node)
    # print('score_diff: ', score_differential * weights[0], 'score_last_cell: ', score_last_placement_in_region * weights[1],  'score_one_empty: ',  score_second_to_last_placement_in_region * weights[2], 'mobility: ', mobility * weights[3], "centrality: ", centrality * weights[4])
    return score_differential * weights[0] + score_last_placement_in_region * weights[1] + score_second_to_last_placement_in_region * weights[2] + mobility * weights[3] + centrality * weights[4]


def weights_at_game_stage(node):
    ''' We want to divide the game into three stages. start (0), mid (1), end (2)'''

    nr_total_squares = node.board.N ** 2
    nr_empty_squares = nr_total_squares - len(node.occupied_squares1) - len(node.occupied_squares2)
    proportion_of_empty_cells = nr_empty_squares / nr_total_squares

    if proportion_of_empty_cells >= 0.7:
        stage = 0
        weights = 0, 0, 0, 2, 2  # explore run and stay central
    elif 0.5 < proportion_of_empty_cells < 0.7:
        stage = 1
        weights = 1, 1, 1, 3, 1  # explore but dont give away easy points, collect easy points and staying central is not as important anymore
    else:  # proportion_of_empty_cells <= 0.3
        stage = 2
        weights = 3, 3, 3, 2, 0  # Only focus on points, if few points can be score use mobility instead

    return stage, weights


def evaluate_last_placement_in_region(node):
    score = 0
    cell_last_move = node.last_move.square

    # Identify if last move finished a region
    cell, nr_completed_regions = find_full_regions_of_cell(node, cell_last_move)  
    
    # For each last cell, determine if it's reachable by the opponent
    if is_not_reachable_by_opponent_and_not_max_points(node, cell, nr_completed_regions):
        score -= 8
    else:  # if last empty cell in region and not reachable by opponent it must be reachable by you
        score += 8
    
    # return score
    if node.current_player == node.my_player:
        return -score
    else:
        return score    
    

def find_full_regions_of_cell(node, cell):
    """
    """
    N = node.board.N
    m, n = node.board.m, node.board.n  # assuming m and n are attributes of SudokuBoard
    
    rows = {r: [] for r in range(N)}
    cols = {c: [] for c in range(N)}
    num_block_rows = N // m
    num_block_cols = N // n
    blocks = {(br, bc): [] for br in range(num_block_rows) for bc in range(num_block_cols)}
    
    # Populate empty cells in each region --> so if only one value in a row, col or block dictionary this means that last one, this is what we check in the next part
    for r in range(N):
        for c in range(N):
            if node.board.get((r, c)) == SudokuBoard.empty:
                rows[r].append((r, c))
                cols[c].append((r, c))
                block = (r // m, c // n)
                blocks[block].append((r, c))
    
    # We want to know how many regions the last move computed
    r, c = cell
    block = (r // m, c // n)
    nr_regions_completed = 0
    if len(rows[r]) == 0:
        nr_regions_completed += 1
    if len(cols[c]) == 0:
        nr_regions_completed += 1
    if len(blocks[block]) == 0:
        nr_regions_completed += 1
    
    return cell, nr_regions_completed
    
    
def is_not_reachable_by_opponent_and_not_max_points(node, cell, nr_completed_regions):
    """
    Determines if a given cell is reachable by the opponent.
    
    @param cell: The cell to check.
    @return: True if reachable by opponent, False otherwise.
    """
    simulated_state = copy.deepcopy(node)
    simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
    playable_squares_opponent = simulated_state.player_squares()
    if playable_squares_opponent is not None:
        # Check if the cell is in opponent's allowed squares
        if cell not in playable_squares_opponent and nr_completed_regions < 3:
            return True
    return False


def evaluate_second_to_last_placement_in_region(node):
    score = 0
    player_squares = node.player_squares()

    # Identify all last cells in their regions
    last_cells = find_last_cells(node, player_squares)
    
    # For each last cell, determine if it's reachable by the opponent
    for cell in last_cells:
        if is_reachable_by_opponent(node, cell):
            score -= 8
        else:  # if last empty cell in region and not reachable by opponent it must be reachable by you
            score += 8
    
    # Given that it is our turn, the opponent put the last move. So we are evaluating the move of the opponent
    # If the opponent put a number in the second to last cell of a region, and it is reachable by you, you want 
    # to negatively evaluate this move by the opponent 

    if node.current_player == node.my_player:
        return -score
    else:
        return score    


def find_last_cells(node, player_squares):
    """
    Identifies cells in player_squares that are the last empty cell in their row, column, or block.
    
    @param player_squares: List of squares where the current player can play.
    @return: List of squares that are the last empty cell in at least one region.
    """
    last_cells = set()
    N = node.board.N
    m, n = node.board.m, node.board.n  # assuming m and n are attributes of SudokuBoard
    
    rows = {r: [] for r in range(N)}
    cols = {c: [] for c in range(N)}
    num_block_rows = N // m
    num_block_cols = N // n
    blocks = {(br, bc): [] for br in range(num_block_rows) for bc in range(num_block_cols)}
    
    # Populate empty cells in each region --> so if only one value in a row, col or block dictionary this means that last one, this is what we check in the next part
    for r in range(N):
        for c in range(N):
            if node.board.get((r, c)) == SudokuBoard.empty:
                rows[r].append((r, c))
                cols[c].append((r, c))
                block = (r // m, c // n)
                blocks[block].append((r, c))
    
    # For each player_square, check if it's the last in any region --> if only one item in row, col, block dictionary then one empty cell in region.
    for cell in player_squares:
        r, c = cell
        block = (r // m, c // n)
        if len(rows[r]) == 1:
            last_cells.add(cell)
        if len(cols[c]) == 1:
            last_cells.add(cell)
        if len(blocks[block]) == 1:
            last_cells.add(cell)
    
    return list(last_cells)
    
    
def is_reachable_by_opponent(node, cell):
    """
    Determines if a given cell is reachable by the opponent.
    
    @param cell: The cell to check.
    @return: True if reachable by opponent, False otherwise.
    """
    simulated_state = copy.deepcopy(node)
    simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
    playable_squares_opponent = simulated_state.player_squares()
    if playable_squares_opponent is not None:
        # Check if the cell is in opponent's allowed squares
        if cell in playable_squares_opponent:
            return True
    return False


def calculate_score_differential(node):
    ''' Calculates the current score differential, so our points - points of opponent. '''

    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]


def positively_evaluate_mobility(node):
    ''' Takes as input the game state and looks how many moves our agent can play. If it is not
    our move we make a null-move, i.e. we copy the GameState and change the current_player, so we
    can check how many adjacient squares we obtain to make a move in the future. '''

    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares())
    
    else: # if it is not our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares())


def evaluate_central_control(node):
    """
    Evaluate the score of cell that are more central on the board
    """
    N = node.board.N

    bullseye_scores = bullseye_centrality_score_division_over_board(N)

    # Now, sum up the scores for the player's controlled cells
    control_score = 0
    occupied_squares = node.occupied_squares()  # or []

    # Add control score when you occupy cells in the middle
    for square in occupied_squares:
        i, j = square
        control_score += bullseye_scores[i][j]
    
    if node.current_player == node.my_player:
        return -control_score
    else:
        return control_score    


def bullseye_centrality_score_division_over_board(N, outer_ring_value=0, ring_value_step=1, central_cell_value_step=2):
    '''awards extra points in rings. Outer ring is zero points, then 1 extra value for ring inside, and 2 extra for center ring/cell'''
    center = N // 2
    max_ring = center  # Number of rings from outer ring to center

    # Create a matrix to hold scores for each cell
    cell_scores = [[0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            # Calculate the ring index for cell (i, j)
            ring = min(i, j, N - 1 - i, N - 1 - j)
            
            if N % 2 == 1 and ring == max_ring:
                # Center cell for odd N
                score = outer_ring_value + ring_value_step * (ring - 1) + central_cell_value_step
            elif N % 2 == 0 and (ring == max_ring - 1 or ring == max_ring):
                # Central cells for even N
                score = outer_ring_value + ring_value_step * (max_ring - 1) + central_cell_value_step
            else:
                score = outer_ring_value + ring_value_step * ring
            
            cell_scores[i][j] = score

    # print board centrality values
    # for row in cell_scores:
    #     print(row)

    return cell_scores
