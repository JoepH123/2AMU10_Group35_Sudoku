import copy
import random
import math
import numpy as np
class MCT_node:
    def __init__(self, gamestate, parent=None, move=None):
        self.my_player = gamestate.current_player
        self.current_player = gamestate.current_player
        self.N = gamestate.board.N
        self.n = gamestate.board.n
        self.m = gamestate.board.m
        self.scores = gamestate.scores
        self.board = self.create_board(gamestate.occupied_squares1, gamestate.occupied_squares2)
        self.parent = parent        # parent Node
        self.move = move            # move that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0  # we will consider "wins" from the perspective of the root player

    def reset_node(self, parent, move):
        self.parent = parent        # parent Node
        self.move = move            # move that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0

    def create_board(self, occupied_squares1, occupied_squares2):
        board = np.zeros((self.N, self.N), dtype=int)

        # Convert lists of tuples into NumPy arrays (if they are not empty)
        if occupied_squares1:
            px, py = np.array(occupied_squares1).T  # Split into x and y arrays
            board[px, py] = 1

        if occupied_squares2:
            ox, oy = np.array(occupied_squares2).T
            board[ox, oy] = 2

        return board

    def get_legal_moves(self):
        """
        Returns a list of (x, y) coordinates for all legal moves for the current player.
        A legal move is any empty cell (value = 0) that is in row 0 (if current_player=1)
        or row N-1 (if current_player=2), OR is adjacent (in any of the 8 directions)
        to at least one cell with the current player.
        """
        N = self.N
        board = self.board
        current_player = self.current_player

        # 1) Identify empty cells
        empty_mask = (board == 0)

        # 2) Build a mask for the "home" row (row 0 for player 1, row N-1 for player 2)
        row_mask = np.zeros((N, N), dtype=bool)
        if current_player == 1:
            row_mask[0, :] = True
        else:
            row_mask[N - 1, :] = True

        # 3) Find all positions occupied by the current_player
        current_positions = np.argwhere(board == current_player)
        if len(current_positions) == 0:
            # If the current player has no pieces on the board yet,
            # only the home row cells are legal (and empty).
            legal_mask = empty_mask & row_mask
            legal_moves = list(zip(*np.where(legal_mask)))
            return legal_moves

        # 4) Vectorized neighbor computation
        directions = np.array([
            [-1, -1], [-1,  0], [-1,  1],
            [ 0, -1],           [ 0,  1],
            [ 1, -1], [ 1,  0], [ 1,  1]
        ])  # shape = (8, 2)

        # Expand current_positions by directions:
        #   current_positions has shape (k, 2)
        #   directions has shape (8, 2)
        # -> neighbors has shape (k, 8, 2)
        neighbors = current_positions[:, None, :] + directions[None, :, :]

        # 5) Filter out-of-bounds neighbors
        valid_x = (neighbors[:, :, 0] >= 0) & (neighbors[:, :, 0] < N)
        valid_y = (neighbors[:, :, 1] >= 0) & (neighbors[:, :, 1] < N)
        valid_mask = valid_x & valid_y

        # Flatten the valid neighbors into a 2D array of shape (num_valid_neighbors, 2)
        valid_neighbors = neighbors[valid_mask]

        # 6) Create a boolean mask of neighbors of current_player
        neighbors_of_current_player = np.zeros((N, N), dtype=bool)
        neighbors_of_current_player[valid_neighbors[:, 0],
        valid_neighbors[:, 1]] = True

        # 7) Combine masks: a legal move is empty AND (in home row OR is neighbor)
        legal_mask = empty_mask & (row_mask | neighbors_of_current_player)

        # 8) Extract coordinates and return
        legal_moves = list(zip(*np.where(legal_mask)))
        return legal_moves

    def is_fully_expanded(self):
        """Check if all possible children (moves) are expanded."""
        if len(self.get_legal_moves())==0 and len(self.children) != 0 and not self.is_terminal():
            return True
        elif len(self.children) == len(self.get_legal_moves()) and len(self.get_legal_moves()) > 0:
            return True
        return False

    def is_terminal(self):
        return np.all(self.board != 0)

    def calculate_score(self, move):
        x, y = move

        m = self.m
        n = self.n

        row_complete = np.all(self.board[x, :] != 0)
        col_complete = np.all(self.board[:, y] != 0)

        region_row = (x // m) * m
        region_col = (y // n) * n

        region = self.board[region_row:region_row + m, region_col:region_col + n]
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

    def best_child(self, c_param=1.5):
        """
        Use UCB1 to select a child node.
        UCB = (wins / visits) + c_param * sqrt(ln(parent_visits) / visits)
        """
        best_value = -float('inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0:
                return child  # if any child is unvisited, pick it immediately
            uct_value = (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif math.isclose(uct_value, best_value):
                best_nodes.append(child)
        best_node = random.choice(best_nodes)
        return best_node  # break ties randomly

    def make_new_child(self, move):
        new_child = copy.deepcopy(self)
        new_child.reset_node(parent=self, move=move)
        new_child.board[move] = self.current_player
        score = new_child.calculate_score(move)
        new_child.scores[new_child.current_player - 1] += score
        if new_child.current_player == 1:
            new_child.current_player = 2
        else:
            new_child.current_player = 1
        return new_child

    def make_new_blocked_child(self):
        new_child = copy.deepcopy(self)
        new_child.reset_node(parent=self, move='blocked')
        if new_child.current_player == 1:
            new_child.current_player = 2
        else:
            new_child.current_player = 1
        return new_child

    def expand(self):
        """Create a new child node for one of the untried moves."""
        tried_moves = [child.move for child in self.children]
        possible_moves = self.get_legal_moves()

        for move in possible_moves:
            if move not in tried_moves:
                child_node = self.make_new_child(move)
                self.children.append(child_node)
                return child_node

        if len(possible_moves) == 0:
            child_node = self.make_new_blocked_child()
            self.children.append(child_node)
            return child_node

    def update(self, result):
        """
        Update node statistics.
        'result' should be +1 if the current player to move in the root state eventually won,
        0 if it was a draw, and -1 if they lost.
        """
        self.visits += 1
        self.wins += result
