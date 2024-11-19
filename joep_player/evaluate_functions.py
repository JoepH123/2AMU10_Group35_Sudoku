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

def evaluate_board(node):
    ''' Combining the evaluation functions '''
    # stage, weights = weights_at_game_stage(node)
    score_differential = calculate_score_differential(node)
    mobility = calculate_mobility(node)
    # control_center = 
    # limit_opponent_mobility = 

    return score_differential * 0.5 + mobility * 0.5


def calculate_score_differential(node):
    ''' Calculates the current score differential, so our points - points of opponent. '''

    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]


def calculate_mobility(node):
    ''' Takes as input the game state and looks how many moves our agent can play. If it is not
    our move we make a null-move, i.e. we copy the GameState and change the current_player, so we
    can check how many adjacient squares we obtain to make a move in the future. '''

    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares())
    
    else: # if it is not our turn  --> what happens when it is not our turn?
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares())
    

def weights_at_game_stage(node):
    ''' We want to divide the game into three stages. start (0), mid (1), end (2)'''

    nr_total_squares = node.board.N ** 2
    nr_empty_squares = nr_total_squares - len(node.occupied_squares1) - len(node.occupied_squares2)
    proportion_of_empty_cells = nr_empty_squares / nr_total_squares

    if proportion_of_empty_cells >= 0.7:
        stage = 0
        weights = 0.25, 0.25, 0.25, 0.25
    elif 0.3 < proportion_of_empty_cells < 0.7:
        stage = 1
        weights = 0.25, 0.25, 0.25, 0.25
    else:  # proportion_of_empty_cells <= 0.3
        stage = 2
        weights = 0.25, 0.25, 0.25, 0.25

    return stage, weights


# def evaluate_central_control(game_state: GameState) -> float:
#     """
#     Evaluates the player's control over the central cells of the board.

#     Returns a higher score for more central control.
#     """
#     N = game_state.board.N
#     center = N // 2
#     central_cells = []

#     # Define central cells (for even N, take the central 2x2 square)
#     if N % 2 == 1:
#         central_cells.append((center, center))
#     else:
#         central_cells.extend([
#             (center - 1, center - 1), (center - 1, center),
#             (center, center - 1), (center, center)
#         ])

#     control_score = 0
#     for square in central_cells:
#         if game_state.board.get(square) == SudokuBoard.empty:
#             if square in game_state.player_squares():
#                 control_score += 1  # Potential to occupy
#         elif square in game_state.occupied_squares():
#             control_score += 2  # Already occupied by player
#     return control_score

    
# def evaluate_opponent_mobility(game_state: GameState) -> float:
#     """
#     Evaluates the opponent's mobility to identify opportunities to limit their moves.

#     Returns a lower score if the opponent has more mobility (since we want to minimize it).
#     """
#     # Simulate switching to the opponent
#     opponent_state = copy.deepcopy(game_state)
#     opponent_state.current_player = 2 if game_state.current_player == 1 else 1

#     opponent_squares = opponent_state.player_squares()
#     if opponent_squares is None:
#         mobility = len(opponent_state.board.empty_squares())
#     else:
#         mobility = len(opponent_squares)
#     return -mobility  # Negative because we want to minimize opponent mobility


# def evaluate_mobility(game_state: GameState) -> float:
#     """
#     Evaluates the mobility of the current player based on the number of valid moves available.

#     Returns a higher score for more mobility.
#     """
#     player_squares = game_state.player_squares()
#     if player_squares is None:
#         # All empty squares are allowed
#         mobility = len(game_state.board.empty_squares())
#     else:
#         mobility = len(player_squares)
#     return mobility


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


