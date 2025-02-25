def unique_controlled_squares(node):
    """
    Calculate the score based on how many unique squares the player can reach
     minus how many unique the opponent can reach.

    Parameters:
    node (Node): The current game state.

    Returns:
    int: The score based on unique squares.
    """
    N = node.board.N
    # find the player squares and make a set of it
    current_player_control_squares = node.player_squares()
    other_player_control_squares = node.other_player_squares()
    current_player_control_squares = set(current_player_control_squares)
    other_player_control_squares = set(other_player_control_squares)

    # Find amount of unique for each player
    unique_current_player = len(current_player_control_squares - other_player_control_squares)/N
    unique_other_player = len(other_player_control_squares - current_player_control_squares)/N

    # assign the score in the correct way
    if node.current_player == node.my_player:
        return unique_current_player - unique_other_player
    else:
        return unique_other_player - unique_current_player

def score_one_empty_in_region(node) -> int:
    """
    Calculate the score based on regions (rows, columns, blocks) that have exactly one empty cell.

    Parameters:
    node (Node): The current game state.

    Returns:
    int: The score based on regions with one empty cell.
    """
    N = node.board.N  # Board size (N x N grid)
    n, m = node.board.n, node.board.m  # Block dimensions

    player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    opponent_occupied = (node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1)

    def get_neighbors(cell):
        """
        Get all valid neighbors of a cell within the board boundaries.
        """
        row, col = cell
        return [(row + dr, col + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                if 0 <= row + dr < N and 0 <= col + dc < N]

    def process_region(cells, current_score):
        """
        Process a region (row, column, or block) and update the score if it has exactly one empty cell.
        """
        empty_cells = [(r, c) for r, c in cells if node.board.get((r, c)) == 0]
        if len(empty_cells) == 1:
            empty_cell = empty_cells[0]
            neighbors = get_neighbors(empty_cell)
            if node.current_player == node.my_player:
                if any(n in opponent_occupied for n in neighbors) and any(n in player_occupied for n in neighbors):
                    current_score += 1
            else:
                if any(n in opponent_occupied for n in neighbors):
                    current_score -= 1
        return current_score

    score_one_empty = 0

    # Process rows, columns, and blocks
    for i in range(N):
        score_one_empty = process_region([(i, c) for c in range(N)], score_one_empty)  # Row
        score_one_empty = process_region([(r, i) for r in range(N)], score_one_empty)  # Column

    for block_row in range(n):
        for block_col in range(m):
            score_one_empty = process_region([(r, c)
                                              for r in range(block_row * m, (block_row + 1) * m)
                                              for c in range(block_col * n, (block_col + 1) * n)], score_one_empty)

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

    # Calculate the center row based on the player's perspective
    center_row = (N / 2) + 0.5 if node.my_player == 1 else (N / 2) - 1.5
    center_col = N // 2  # Center column is the middle column

    row_weight = 0.99  # Weight for row distance
    col_weight = 0.01  # Weight for column distance

    def weighted_distance(row: int, col: int) -> float:
        """
        Calculate the weighted distance of a cell to the center of the board.
        """
        return (((row - center_row) * row_weight) ** 2 + ((col - center_col) * col_weight) ** 2) ** 0.5

    # Calculate proximity score for player 1
    player1_score = sum(max(0, N - weighted_distance(row, col)) for row, col in node.occupied_squares1) / N

    # Calculate proximity score for player 2
    player2_score = sum(max(0, N - weighted_distance(row, col)) for row, col in node.occupied_squares2) / N

    # Return the evaluation from the perspective of the current player
    return player1_score - player2_score if node.my_player == 1 else player2_score - player1_score


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


def punish_corner(node):
    """
    Penalizes a player if they occupy specific corner squares on the board.

    Args:
        node: An object representing the game state, including board size,
              player positions, and current player.

    Returns:
        int: Penalty score (-10 if the corner is occupied, 0 otherwise).
    """
    m = node.board.m
    n = node.board.n
    if node.my_player == 1:
        # Get the occupied squares for player 1
        player_occupied = node.occupied_squares1
        score = 0
        for row in range(m-1):
            for col in range(n-1):
                if (row, col) in player_occupied:  # Check top-left corner
                    score = -10
        return score

    if node.my_player == 2:
        # Get the occupied squares for player 2
        N = node.board.N  # Board size
        player_occupied = node.occupied_squares2
        score = 0
        for row in range(N-m+1, N):
            for col in range(n-1):
                if (row, col) in player_occupied:  # Check bottom-left corner
                    score = -10
        return score


def evaluate_node(node) -> float:
    """
    Evaluate the given node and return a score.

    Parameters:
    node (Node): The current game state.

    Returns:
    float: The evaluation score.
    """
    N = node.board.N
    score_diff_game = calculate_score_difference(node)
    score_one_empty = score_one_empty_in_region(node)

    if node.game_phase == 'late':
        # if board size smaller than 3x3 also use punish_corner in evaluation of late game
        punish_corner_score = punish_corner(node) if N<9 else 0
        eval_func = score_diff_game + score_one_empty + punish_corner_score

    else:
        punish_corner_score = punish_corner(node)
        score_center = calc_score_center_moves(node)
        unique_control = unique_controlled_squares(node)
        eval_func = score_diff_game + score_one_empty + punish_corner_score + score_center + unique_control

    return eval_func

