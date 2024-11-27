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

# Simple minimax evaluation function only based on maximizing the score

def evaluate_node(node):
    ''' Combining the evaluation functions '''
    score_differential = calculate_score_differential(node)

    return score_differential


def calculate_score_differential(node):
    ''' Calculates the current score differential, so our points - points of opponent. '''

    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]
