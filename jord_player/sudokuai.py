#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from .Jordseval import evaluate_board


class NodeGameState(GameState):
    def __init__(self, game_state, root_move=None, last_move=None, my_player=None):
        """
        Initialize a NodeGameState by copying the given game_state and adding extra attributes.

        Parameters:
        - game_state (GameState): The game state to copy.
        - root_move (Move, optional): The initial move leading to this node.
        - last_move (Move, optional): The last move made.
        - my_player (int, optional): The AI's player number.
        """
        self.__dict__ = game_state.__dict__.copy()
        self.root_move = root_move
        self.last_move = last_move
        self.my_player = my_player

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    def __init__(self):
        """Initialize the SudokuAI by calling the superclass initializer."""
        super().__init__()
        self.killer_moves = {}  # Dictionary to store killer moves for each depth
        self.nodes_explored = 0  # Add this line to initialize the counter
        self.killer_move_cutoffs = 0  # Teller voor killer move cutoffs
        self.total_cutoffs = 0  

    def evaluate(self, node):
        """
        Evaluate the heuristic value of a node.

        Parameters:
        - node (NodeGameState): The game state node to evaluate.

        Returns:
        - float: The heuristic value of the node.
        """
        return evaluate_board(node)

    def is_valid_move_possible(self, node):
        """
        Check if there is at least one valid move possible for the current player.

        Parameters:
        - node (NodeGameState): The current game state.

        Returns:
        - bool: True if at least one valid move is possible, False otherwise.
        """
        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions

        def is_value_valid(row, col, value):
            """
            Check if placing 'value' at position (row, col) is valid according to Sudoku rules.

            Parameters:
            - row (int): Row index.
            - col (int): Column index.
            - value (int): The value to place.

            Returns:
            - bool: True if the move is valid, False otherwise.
            """
            # Precompute block starting indices
            block_start_row = (row // m) * m
            block_start_col = (col // n) * n

            for i in range(N):
                # Check row
                if board.get((row, i)) == value:
                    return False

                # Check column
                if board.get((i, col)) == value:
                    return False

                # Check block
                block_row = block_start_row + i // n
                block_col = block_start_col + i % n
                if board.get((block_row, block_col)) == value:
                    return False

            return True

        # Iterate over all cells on the board
        for row in range(N):
            for col in range(N):
                if board.get((row, col)) == SudokuBoard.empty:  # Empty cell
                    # Check if any value (1 to N) can be placed in this cell
                    for value in range(1, N + 1):
                        if is_value_valid(row, col, value):
                            return True  # Found at least one valid move

        return False  # No valid moves found

    def is_terminal(self, node):
        """
        Determine if the game has reached a terminal state (no valid moves left).

        Parameters:
        - node (NodeGameState): The current game state.

        Returns:
        - bool: True if the game is over, False otherwise.
        """
        return not self.is_valid_move_possible(node)

    def respects_rule_C0(self, node, row, col, value):
        """
        Check if placing 'value' at (row, col) respects the basic Sudoku rules.

        Parameters:
        - node (NodeGameState): The current game state.
        - row (int): Row index.
        - col (int): Column index.
        - value (int): The value to place.

        Returns:
        - bool: True if the move respects the rules, False otherwise.
        """
        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions

        # Precompute block starting indices
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n

        for i in range(N):
            # Check row
            if board.get((row, i)) == value:
                return False

            # Check column
            if board.get((i, col)) == value:
                return False

            # Check block
            block_row = block_start_row + i // n
            block_col = block_start_col + i % n
            if board.get((block_row, block_col)) == value:
                return False

        return True

    def playable_square(self, node, i, j):
        """
        Controleer of de cel op positie (i, j) speelbaar is (d.w.z. leeg, in de speler zijn vakjes, en ten minste één legale waarde heeft).

        Parameters:
        - node (NodeGameState): De huidige speltoestand.
        - i (int): Rij-index.
        - j (int): Kolom-index.

        Returns:
        - bool: True als de cel speelbaar is, anders False.
        """
        return (node.board.get((i, j)) == SudokuBoard.empty
                and (i, j) in node.player_squares()
                and self.get_legal_values(node, i, j))


    def calculate_move_score(self, node, move):
        """
        Calculate the score obtained by making a particular move.

        Parameters:
        - node (NodeGameState): The current game state.
        - move (Move): The move to evaluate.

        Returns:
        - int: The score obtained from the move.
        """
        board = node.board
        N = board.N  # Size of the grid (N = n * m)
        n, m = board.n, board.m  # Block dimensions
        row, col = move.square  # Get the row and column of the move

        # Precompute block starting indices
        block_start_row = (row // m) * m
        block_start_col = (col // n) * n

        def is_region_complete(values):
            """
            Check if a region (row, column, or block) is complete.

            Parameters:
            - values (list of int): The values in the region.

            Returns:
            - bool: True if the region is complete, False otherwise.
            """
            return (len(values) == N
                    and len(set(values)) == N
                    and all(value != SudokuBoard.empty for value in values))

        # Collect values in the row, column, and block
        row_values = [board.get((row, i)) for i in range(N)]
        col_values = [board.get((i, col)) for i in range(N)]
        block_values = [
            board.get((block_start_row + (i // n), block_start_col + (i % n)))
            for i in range(N)
        ]

        # Count completed regions
        completed_regions = sum([
            is_region_complete(row_values),
            is_region_complete(col_values),
            is_region_complete(block_values)
        ])

        # Return score based on completed regions
        if completed_regions == 1:
            return 1  # 1 point for completing 1 region
        elif completed_regions == 2:
            return 3  # 3 points for completing 2 regions
        elif completed_regions == 3:
            return 7  # 7 points for completing all 3 regions
        else:
            return 0  # No regions completed


    def calculate_move_heuristic(self, node, move):
        """
        Assign a heuristic score to a move for move ordering.

        Parameters:
        - node (NodeGameState): The current game state.
        - move (Move): The move to evaluate.

        Returns:
        - float: The heuristic score of the move.
        """
        # Prioritize moves that complete regions, as they yield higher scores
        move_score = self.calculate_move_score(node, move)

        # Additional heuristics can be added here
        # For example, prioritize moves that:
        # - Reduce opponent's mobility
        # - Occupy central squares
        # - Block opponent's potential high-scoring moves

        # For simplicity, we'll use the move score as the heuristic
        return move_score
    

    def get_legal_values(self, node, row, col):
        """
        Verkrijg alle legale waarden voor een gegeven cel, exclusief taboo moves.

        Parameters:
        - node (NodeGameState): De huidige speltoestand.
        - row (int): Rij-index.
        - col (int): Kolom-index.

        Returns:
        - list van int: Een lijst met legale waarden voor de cel.
        """
        N = node.board.N
        legal_values = []
        for value in range(1, N + 1):
            if (self.respects_rule_C0(node, row, col, value)
                and TabooMove((row, col), value) not in node.taboo_moves):
                legal_values.append(value)
        return legal_values



    def get_all_moves(self, node):
        """
        Genereer een lijst van zetten door één willekeurige legale waarde te selecteren voor elke lege cel.

        Parameters:
        - node (NodeGameState): De huidige speltoestand.

        Returns:
        - lijst van Move: Een lijst van zetten met één willekeurige legale waarde per lege cel.
        """
        N = node.board.N
        all_moves = []
        for i in range(N):
            for j in range(N):
                if (node.board.get((i, j)) == SudokuBoard.empty
                    and (i, j) in node.player_squares()):
                    legal_values = self.get_legal_values(node, i, j)
                    if legal_values:
                        # Kies een willekeurige legale waarde
                        value = random.choice(legal_values)
                        move = Move((i, j), value)
                        all_moves.append(move)
        return all_moves



    def apply_move(self, node, move):
        """
        Apply a move to a node and return the resulting child node.

        Parameters:
        - node (NodeGameState): The current game state.
        - move (Move): The move to apply.

        Returns:
        - NodeGameState: The new game state after applying the move.
        """
        new_node = copy.deepcopy(node)
        new_node.last_move = move
        new_node.board.put(move.square, move.value)
        new_node.moves.append(move)
        # Update the score for the current player
        score = self.calculate_move_score(new_node, move)
        new_node.scores[new_node.current_player - 1] += score
        # Update occupied squares and switch player
        if new_node.current_player == 1:
            new_node.occupied_squares1.append(move.square)
            new_node.current_player = 2
        else:
            new_node.occupied_squares2.append(move.square)
            new_node.current_player = 1
        return new_node

    def minimax(self, node, depth, is_maximizing_player, alpha, beta):
        self.nodes_explored += 1
        """
        Perform the Minimax algorithm with Alpha-Beta pruning.

        Parameters:
        - node (NodeGameState): The current game state.
        - depth (int): The depth limit for the search.
        - is_maximizing_player (bool): True if it's the maximizing player's turn.
        - alpha (float): Alpha value for pruning.
        - beta (float): Beta value for pruning.

        Returns:
        - float: The evaluated score of the node.
        """
        # Base case: If maximum depth is reached or game state is terminal
        if depth == 0 or self.is_terminal(node):
            return self.evaluate(node)

        if is_maximizing_player:
            max_eval = float('-inf')
            moves = self.get_all_moves(node)

            # Check for killer move at this depth
            killer_move = self.killer_moves.get(depth)
            if killer_move and killer_move in moves:
                moves.remove(killer_move)
                moves.insert(0, killer_move)  # Try killer move first
                #print('killer move is',killer_move)

            for move in moves:
                child = self.apply_move(node, move)
                eval_score = self.minimax(child, depth - 1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    if eval_score >= beta:
  
                        # Update killer move
                        self.killer_moves[depth] = move
                        return max_eval
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return max_eval
        
        else:
            min_eval = float('inf')
            moves = self.get_all_moves(node)

            # Check for killer move at this depth
            killer_move = self.killer_moves.get(depth)
            if killer_move and killer_move in moves:
                moves.remove(killer_move)
                moves.insert(0, killer_move)  # Try killer move first

            for move in moves:
                child = self.apply_move(node, move)
                eval_score = self.minimax(child, depth - 1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    if eval_score <= alpha:
                        # Update killer move
                        self.killer_moves[depth] = move
                        return min_eval
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            return min_eval

    def compute_best_move(self, game_state: GameState) -> None:
        root_node = NodeGameState(game_state)
        root_node.my_player = root_node.current_player
        max_depth = 30  # Adjust as needed

        self.nodes_explored = 0  # Reset the counter before the search

        best_move = None
        best_value = float('-inf')

        for depth in range(1, max_depth + 1):
            self.killer_moves = {}  # Reset killer moves for each depth
            moves = self.get_all_moves(root_node)
            for move in moves:
                child = self.apply_move(root_node, move)
                move_value = self.minimax(child, depth - 1, False, float('-inf'), float('inf'))
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
                    self.propose_move(best_move)
            print(f'Depth {depth} search complete.')
            print(f'Nodes explored at depth {depth}: {self.nodes_explored}')
            self.nodes_explored = 0  # Reset if tracking per depth

