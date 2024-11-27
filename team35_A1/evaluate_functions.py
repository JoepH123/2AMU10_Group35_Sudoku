import copy

def calculate_mobility(node) -> float:
    """
    Calculate the mobility score for the current node.

    Parameters:
    node (Node): The current game state.

    Returns:
    float: The mobility score, representing the number of available moves normalized by board size.
    """
    N = node.board.N
    if node.my_player == node.current_player:  # If it is our turn
        return len(node.player_squares()) / N
    else:  # If it is not our turn
        # Simulate the state with our player to move
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player  # Swap current player
        return len(simulated_state.player_squares()) / N


def score_one_empty_in_region(node) -> int:
    """
    Calculate the score based on regions (rows, columns, blocks) that have exactly one empty cell.

    Parameters:
    node (Node): The current game state.

    Returns:
    int: The score based on regions with one empty cell.
    """
    my_player = node.my_player
    N = node.board.N  # Board size (N x N grid)
    n, m = node.board.n, node.board.m  # Block dimensions

    def get_neighbors(row: int, col: int):
        """
        Get all neighboring cells (including diagonals) for a given cell within the board boundaries.

        Parameters:
        row (int): Row index.
        col (int): Column index.

        Returns:
        list of tuples: List of (row, col) indices of neighboring cells.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [
            (row + dr, col + dc)
            for dr, dc in directions
            if 0 <= row + dr < N and 0 <= col + dc < N
        ]

    def empty_cells_in_region(cells) -> int:
        """
        Count the number of empty cells in a given list of cells.

        Parameters:
        cells (list of tuples): List of (row, col) indices.

        Returns:
        int: Number of empty cells.
        """
        return sum(1 for r, c in cells if node.board.get((r, c)) == 0)

    def occupied_neighbors(cells, occupied_squares) -> bool:
        """
        Check if any empty cell in the given cells has a neighbor in occupied_squares.

        Parameters:
        cells (list of tuples): List of (row, col) indices.
        occupied_squares (set): Set of (row, col) indices occupied by a player.

        Returns:
        bool: True if any empty cell has an occupied neighbor, False otherwise.
        """
        for r, c in cells:
            if node.board.get((r, c)) == 0:  # Only check empty cells
                for neighbor in get_neighbors(r, c):
                    if neighbor in occupied_squares:
                        return True
        return False

    def get_row_cells(row: int):
        """
        Get all cell indices in a given row.

        Parameters:
        row (int): Row index.

        Returns:
        list of tuples: List of (row, col) indices.
        """
        return [(row, c) for c in range(N)]

    def get_col_cells(col: int):
        """
        Get all cell indices in a given column.

        Parameters:
        col (int): Column index.

        Returns:
        list of tuples: List of (row, col) indices.
        """
        return [(r, col) for r in range(N)]

    def get_block_cells(block_row: int, block_col: int):
        """
        Get all cell indices in a given block.

        Parameters:
        block_row (int): Block row index.
        block_col (int): Block column index.

        Returns:
        list of tuples: List of (row, col) indices.
        """
        return [
            (r, c)
            for r in range(block_row * m, (block_row + 1) * m)
            for c in range(block_col * n, (block_col + 1) * n)
        ]

    # Define players' occupied squares
    player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )
    opponent_occupied = (
        node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1
    )

    score_one_empty = 0

    def process_region(cells):
        """
        Process a region (row, column, or block) and update the score if it has exactly one empty cell.

        Parameters:
        cells (list of tuples): List of (row, col) indices in the region.
        """
        nonlocal score_one_empty
        if empty_cells_in_region(cells) == 1:
            if node.current_player == node.my_player:
                if occupied_neighbors(cells, opponent_occupied) and occupied_neighbors(
                    cells, player_occupied
                ):
                    score_one_empty += 1
            else:
                if occupied_neighbors(cells, opponent_occupied):
                    score_one_empty -= 1

    # Check rows and columns
    for i in range(N):
        process_region(get_row_cells(i))
        process_region(get_col_cells(i))

    # Check blocks
    for block_row in range(n):
        for block_col in range(m):
            process_region(get_block_cells(block_row, block_col))

    return score_one_empty


def calc_score_center_moves(node) -> float:
    """
    Calculate a score based on the proximity of the player's occupied squares to the center of the board.

    Parameters:
    node (Node): The current game state.

    Returns:
    float: The score difference between the current player and the opponent.
    """
    N = node.board.N  # Board size (N x N grid)
    center_row = (N + 1) / 2  # Center point (e.g., 4.5 for a 9x9 grid)
    center_col = (N - 1) / 2
    def distance_to_center(row: int, col: int) -> float:
        """
        Calculate the weighted distance of a cell to the center of the board.

        Parameters:
        row (int): Row index.
        col (int): Column index.

        Returns:
        float: Weighted distance to the center.
        """
        row_weight = 0.99  # Weight for row distance
        col_weight = 0.01  # Weight for column distance
        return ((row_weight * (row - center_row)) ** 2 + (col_weight * (col - center_col)) ** 2) ** 0.5

    def calculate_player_score(occupied_squares) -> float:
        """
        Calculate the proximity score for a player based on their occupied squares.

        Parameters:
        occupied_squares (list of tuples): List of (row, col) indices occupied by the player.

        Returns:
        float: The proximity score.
        """
        proximity_score = 0
        for row, col in occupied_squares:
            proximity_score += max(0, N - distance_to_center(row, col))  # Reward closer cells
        return proximity_score / N

    # Get scores for both players based on their occupied squares
    player1_score = calculate_player_score(node.occupied_squares1)
    player2_score = calculate_player_score(node.occupied_squares2)

    # Return the evaluation from the perspective of the current player
    if node.my_player == 1:
        return player1_score - player2_score
    else:
        return player2_score - player1_score


def calculate_score_difference(node) -> float:
    """
    Calculate the score difference between the current player and the opponent.

    Parameters:
    node (Node): The current game state.

    Returns:
    float: The score difference.
    """
    my_index = node.my_player - 1
    opponent_index = 1 - my_index
    return node.scores[my_index] - node.scores[opponent_index]


def evaluate_node(node) -> float:
    """
    Evaluate the given node and return a heuristic score.

    Parameters:
    node (Node): The current game state.

    Returns:
    float: The evaluation score.
    """
    # Calculate the score difference between players
    score_diff_game = calculate_score_difference(node)

    # Calculate score based on proximity to the center
    score_center = calc_score_center_moves(node)

    # Determine the number of occupied squares by the current player
    num_occupied = len(node.occupied_squares1) if node.my_player == 1 else len(node.occupied_squares2)

    # Early game evaluation
    if num_occupied <= node.board.n-1:
        eval_func = score_diff_game + 2 * score_center
        return eval_func

    # Mid to late game evaluation
    score_one_empty = score_one_empty_in_region(node)
    score_mobility = calculate_mobility(node)

    eval_func = score_diff_game + score_mobility + score_one_empty

    return eval_func

