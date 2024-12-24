import copy
import random
import math
import numpy as np
class MCT_node:
    def __init__(self, gamestate, parent=None, move=None):
        self.my_player = gamestate.current_player
        self.current_player = gamestate.current_player
        self.occupied_squares1 = gamestate.occupied_squares1
        self.occupied_squares2 = gamestate.occupied_squares2
        self.N = gamestate.board.N
        self.n = gamestate.board.n
        self.m = gamestate.board.m
        self.scores = gamestate.scores
        self.board = self.create_board()
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

    def create_board(self):
        board = np.zeros((self.N, self.N), dtype=int)

        # Convert lists of tuples into NumPy arrays (if they are not empty)
        if self.occupied_squares1:
            px, py = np.array(self.occupied_squares1).T  # Split into x and y arrays
            board[px, py] = 1

        if self.occupied_squares2:
            ox, oy = np.array(self.occupied_squares2).T
            board[ox, oy] = 2

        return board

    def get_legal_moves(self):
        """
        Returns a list of (x, y) coordinates for all legal moves for current player.
        A legal move is any empty cell (value = 0) that is in row 0
        OR adjacent (in any of the 8 directions) to at least one cell with current player.
        """
        N = self.N  # assuming square board: NxN
        current_player = self.current_player
        # Mark which cells are neighbors of a cell with a '1'
        neighbors_of_current_player = np.zeros((N, N), dtype=bool)

        # Directions (dx, dy) to cover 8 neighbors (including diagonals)
        directions = [(-1, -1), (-1,  0), (-1,  1),
                      ( 0, -1),           ( 0,  1),
                      ( 1, -1), ( 1,  0), ( 1,  1)]

        # For each cell that contains 1, mark its neighbors
        current_player_positions = np.argwhere(self.board == current_player)  # array of [x,y] where board[x,y] == 1
        for x, y in current_player_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors_of_current_player[nx, ny] = True

        # Build a boolean mask of empty cells
        empty_mask = (self.board == 0)

        # Build a boolean mask for row 0
        row0_mask = np.zeros((N, N), dtype=bool)
        if current_player==1:
            row0_mask[0, :] = True
        else:
            row0_mask[N-1, :] = True

        # A legal move must be empty AND (in row 0 OR neighbor_of_1)
        legal_mask = empty_mask & (row0_mask | neighbors_of_current_player)

        # Extract (x, y) coordinates from the mask
        legal_moves = list(zip(*np.where(legal_mask)))

        return legal_moves

    def is_fully_expanded(self):
        """Check if all possible children (moves) are expanded."""
        if len(self.children) == len(self.get_legal_moves()):
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

    def best_child(self, c_param=2):
        """
        Use UCB1 to select a child node.
        UCB = (wins / visits) + c_param * sqrt(ln(parent_visits) / visits)
        """
        best_value = -float('inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0:
                return child  # if any child is unvisited, pick it immediately
            uct_value = (child.wins / child.visits) + c_param * math.sqrt(
                math.log(self.visits) / child.visits
            )
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif math.isclose(uct_value, best_value):
                best_nodes.append(child)

        return random.choice(best_nodes)  # break ties randomly

    def make_new_child(self, move):
        new_child = copy.deepcopy(self)
        new_child.reset_node(parent=self, move=move)
        new_child.board[move] = self.current_player
        new_child.occupied_squares1.append(move) if new_child.current_player == 1 else new_child.occupied_squares2.append(move)
        score = new_child.calculate_score(move)
        new_child.scores[new_child.current_player - 1] += score
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
        # If no moves are left, return None
        return None

    def update(self, result):
        """
        Update node statistics.
        'result' should be +1 if the current player to move in the root state eventually won,
        0 if it was a draw, and -1 if they lost.
        """
        self.visits += 1
        self.wins += result





