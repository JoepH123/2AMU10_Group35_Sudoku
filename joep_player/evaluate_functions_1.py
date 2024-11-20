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
    score_last_placement_in_region = evaluate_last_placement_in_region(node)
    mobility = positively_evaluate_mobility(node)
    centrality = evaluate_central_control(node)
    # opp_mobility = negatively_evaluate_opponent_mobility(node)  --> does not seem to work well, keeps all number in the back line
    # limit_opponent_mobility = 
    # check if last move completed a region, if this is the case, and the opponent could reach this cell than reward heavily, if opponent could not reach the cell punish heavily
    # We want to postpone finishing regions if opponent cannot finish it

    # return score_differential * weights[0] + score_second_to_last_placement_in_region * weights[1] + mobility * weights[2] + centrality * weights[3]
    return score_differential * 0.2 + score_last_placement_in_region * 0.2 + score_second_to_last_placement_in_region * 0.2 + mobility * 0.2 + centrality * 0.2
    # return score_differential * weights[0] + mobility * weights[1] + opp_mobility * weights[2] + centrality * weights[3]
    # return score_differential * weights[0] + centrality * weights[3] + opp_mobility * weights[2]
    # return opp_mobility


def evaluate_last_placement_in_region(node):
    score = 0
    cell_last_move = node.last_move.square

    # Identify if last move finished a region
    cell, nr_completed_regions = find_full_regions_of_cell(node, cell_last_move)  
    
    # For each last cell, determine if it's reachable by the opponent
    if is_not_reachable_by_opponent_and_not_max_points(node, cell, nr_completed_regions):
        score -8
    else:  # if last empty cell in region and not reachable by opponent it must be reachable by you
        score += 8
    
    return score
    # if node.current_player == node.my_player:
    #     return -score
    # else:
    #     return score    
    

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
    # print("check if reachable")
    # print(cell)
    if playable_squares_opponent is not None:
        # Check if the cell is in opponent's allowed squares
        if cell not in playable_squares_opponent and nr_completed_regions < 3:
            # print("reachable")
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
    return score

    # if node.current_player == node.my_player:
    #     return -score
    # else:
    #     return score    


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
    # print("check if reachable")
    # print(cell)
    if playable_squares_opponent is not None:
        # Check if the cell is in opponent's allowed squares
        if cell in playable_squares_opponent:
            # print("reachable")
            return True
    # else:
        # print("not reachable")
    return False


def weights_at_game_stage(node):
    ''' We want to divide the game into three stages. start (0), mid (1), end (2)'''

    nr_total_squares = node.board.N ** 2
    nr_empty_squares = nr_total_squares - len(node.occupied_squares1) - len(node.occupied_squares2)
    proportion_of_empty_cells = nr_empty_squares / nr_total_squares

    if proportion_of_empty_cells >= 0.7:
        stage = 0
        weights = 0.1, 0.1, 0.4, 0.4
    elif 0.3 < proportion_of_empty_cells < 0.7:
        stage = 1
        weights = 0.1, 0.3, 0.3, 0.3
    else:  # proportion_of_empty_cells <= 0.3
        stage = 2
        weights = 0.4, 0.4, 0.1, 0.1

    return stage, weights


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

    
def negatively_evaluate_opponent_mobility(node):
    """
    Evaluates the opponent's mobility to identify opportunities to limit their moves.

    Returns a lower score if the opponent has more mobility (since we want to minimize it).
    """
    if node.my_player == node.current_player: # if it is our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our opponent to move
        return -len(simulated_state.player_squares())
    
    else: # if it is not our turn
        return -len(node.player_squares())


def evaluate_central_control(node):
    """
    Evaluate the score of cell that are more central on the board
    """
    N = node.board.N

    # cell_scores = bullseye_centrality_score_division_over_board(N)
    # cell_scores = corridor_centrality_score_division_over_board(N)

    bullseye_scores = bullseye_centrality_score_division_over_board(N)
    corridor_scores = corridor_centrality_score_division_over_board(N)
    combined_scores = [[bullseye_scores[i][j] + corridor_scores[i][j] for j in range(N)] for i in range(N)]

    # print board centrality values
    # for row in combined_scores:
    #     print(row)

    # Now, sum up the scores for the player's controlled cells
    control_score = 0
    occupied_squares = node.occupied_squares()  # or []

    # Add control score when you occupy cells in the middle
    for square in occupied_squares:
        i, j = square
        control_score += combined_scores[i][j]

    return control_score


def corridor_centrality_score_division_over_board(N, middle_value=2):
    '''Awards points to two or three central columns, and zero to other columns'''
    # Initialize the scoring matrix with zeros
    cell_scores = [[0 for _ in range(N)] for _ in range(N)]
    
    # Determine middle columns
    if N % 2 == 1:
        # Odd N, middle columns are at N//2 -1, N//2, N//2 + 1
        middle_cols = [N // 2 - 1, N // 2, N // 2 + 1]
    else:
        # Even N, middle columns are at N//2 - 1 and N//2
        middle_cols = [N // 2 - 1, N // 2]
    
    # Assign the middle_value to the middle columns
    for i in range(N):
        for j in middle_cols:
            cell_scores[i][j] = middle_value

    # print board centrality values
    # for row in cell_scores:
    #     print(row)
    
    return cell_scores


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


# def evaluate_self_connectivity(game_state: GameState) -> float:
#     """
#     Evaluates the connectivity of the current player's own accessible squares.
    
#     Returns a lower score when the player's accessible squares are fragmented.
#     """
#     player_squares = game_state.player_squares()
    
#     if not player_squares:
#         # No moves available; extremely bad state
#         return -float('inf')
    
#     # Build a graph of player's accessible squares
#     graph = build_accessible_squares_graph(game_state, player_squares)
    
#     # Find connected components
#     connected_components = find_connected_components(graph, player_squares)
    
#     # Evaluate fragmentation
#     num_components = len(connected_components)
#     largest_component_size = max(len(component) for component in connected_components)
    
#     # Lower score for more components and smaller largest component
#     fragmentation_score = - (num_components * 10 - largest_component_size)
#     return fragmentation_score


# import copy
# from collections import deque

# def evaluate_opponent_connectivity(game_state: GameState) -> float:
#     """
#     Evaluates the degree to which the opponent's allowed squares are fragmented.
    
#     Returns a higher score when the opponent's accessible squares are more disconnected.
#     """
#     opponent_state = copy.deepcopy(game_state)
#     opponent_state.current_player = 2 if game_state.current_player == 1 else 1
#     opponent_squares = opponent_state.player_squares()
    
#     if not opponent_squares:
#         # No moves available to opponent; maximum disconnection
#         return float('inf')
    
#     # Build a graph of opponent's accessible squares
#     graph = build_accessible_squares_graph(opponent_state, opponent_squares)
    
#     # Find connected components
#     connected_components = find_connected_components(graph, opponent_squares)
    
#     # Evaluate fragmentation
#     num_components = len(connected_components)
#     largest_component_size = max(len(component) for component in connected_components)
    
#     # Higher score for more components and smaller largest component
#     fragmentation_score = num_components * 10 - largest_component_size
#     return fragmentation_score


# def build_accessible_squares_graph(game_state: GameState, accessible_squares: List[Square]) -> Dict[Square, List[Square]]:
#     """
#     Builds a graph where nodes are accessible squares for the opponent, and edges connect adjacent squares.
#     """
#     graph = {square: [] for square in accessible_squares}
#     N = game_state.board.N
    
#     for square in accessible_squares:
#         row, col = square
#         # Check all adjacent squares (orthogonally and diagonally)
#         for dr in (-1, 0, 1):
#             for dc in (-1, 0, 1):
#                 if dr == 0 and dc == 0:
#                     continue
#                 r, c = row + dr, col + dc
#                 neighbor = (r, c)
#                 if 0 <= r < N and 0 <= c < N and neighbor in accessible_squares:
#                     graph[square].append(neighbor)
#     return graph


# def find_connected_components(graph: Dict[Square, List[Square]], nodes: List[Square]) -> List[List[Square]]:
#     """
#     Finds connected components in the graph using BFS.
#     """
#     visited = set()
#     components = []
    
#     for node in nodes:
#         if node not in visited:
#             component = []
#             queue = deque([node])
#             visited.add(node)
#             while queue:
#                 current = queue.popleft()
#                 component.append(current)
#                 for neighbor in graph[current]:
#                     if neighbor not in visited:
#                         visited.add(neighbor)
#                         queue.append(neighbor)
#             components.append(component)
#     return components


