import random
from .MCT_functions import *

def make_rollout_move(node, move):
    node.board[move] = node.current_player
    score = node.calculate_score(move)
    node.scores[node.current_player - 1] += score
    if node.current_player == 1:
        node.current_player = 2
    else:
        node.current_player = 1
    return node


def rollout(node):
    """
    Simulate a random play-out from the given state and return the outcome
    from the perspective of 'state.current_player'.
    """
    rollout_node = copy.deepcopy(node)
    while not rollout_node.is_terminal():
        moves = rollout_node.get_legal_moves()
        if len(moves)>0:
            move = random.choice(moves)
            rollout_node = make_rollout_move(rollout_node, move)

    winner = get_winner(rollout_node)
    return winner  # +1 (Player1), -1 (Player2), or 0 (draw)

def get_winner(node):
    score_p1, score_p2 = node.scores
    if score_p1 > score_p2:
        return +1   # Player 1 wins
    elif score_p2 > score_p1:
        return -1   # Player 2 wins
    else:
        return 0    # Draw
