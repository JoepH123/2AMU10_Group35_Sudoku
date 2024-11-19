import copy

def score_block_occupation(node):
    """
    Evaluate the game state and give +1 for each block where the current player
    has at least one occupied square, excluding blocks adjacent to the player's starting row.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the number of blocks occupied by the current player.
    """

    N = node.board.N  # Board size (N x N grid)
    n, m = node.board.n, node.board.m  # Block dimensions

    # Determine the player's starting row
    starting_row = 0 if node.my_player == 1 else N - 1

    # Identify rows adjacent to the starting row
    adjacent_rows = {starting_row}
    if starting_row - 1 >= 0:
        adjacent_rows.add(starting_row - 1)
    if starting_row + 1 < N:
        adjacent_rows.add(starting_row + 1)

    # Identify blocks adjacent to the starting row
    adjacent_blocks = set()
    for r in adjacent_rows:
        block_row = r // m
        for block_col in range(n):
            adjacent_blocks.add((block_row, block_col))

    # Get the current player's occupied squares
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    # Helper to get all squares in a block
    def get_block_cells(block_row, block_col):
        return [
            (r, c)
            for r in range(block_row * m, (block_row + 1) * m)
            for c in range(block_col * n, (block_col + 1) * n)
        ]

    # Initialize the score
    score = 0

    # Iterate over all blocks
    for block_row in range(n):
        for block_col in range(m):
            # Skip blocks adjacent to the starting row
            if (block_row, block_col) in adjacent_blocks:
                continue

            block_cells = get_block_cells(block_row, block_col)

            # Check if the current player occupies at least one square in this block
            if any(cell in current_player_occupied for cell in block_cells):
                score += 1

    return score



def calculate_mobility(node):
    N = node.board.N
    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares()) / N

    else: # if it is not our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares()) / N


def score_one_empty_in_region(node):

    my_player = node.my_player
    N = node.board.N  # Board size (N x N grid)
    n, m = node.board.n, node.board.m  # Block dimensions

    def get_neighbors(row, col):

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1) ]
        return [
            (row + dr, col + dc)
            for dr, dc in directions
            if 0 <= row + dr < N and 0 <= col + dc < N]

    def empty_cells_in_region(cells):

        return sum(1 for r, c in cells if node.board.get((r, c)) == 0)

    def occupied_neighbors(cells, occupied_squares):

        for r, c in cells:
            if node.board.get((r, c)) == 0:  # Only check empty cells
                for neighbor in get_neighbors(r, c):
                    if neighbor in occupied_squares:
                        return True
        return False

    def get_row_cells(row):

        return [(row, c) for c in range(N)]

    def get_col_cells(col):

        return [(r, col) for r in range(N)]

    def get_block_cells(block_row, block_col):

        return [(r, c)
                for r in range(block_row * m, (block_row + 1) * m)
                for c in range(block_col * n, (block_col + 1) * n)]

    # Define players' occupied squares
    player_occupied = (node.occupied_squares1 if node.current_player == 1 else node.occupied_squares2)
    opponent_occupied = (node.occupied_squares2 if node.current_player == 1 else node.occupied_squares1)

    # Initialize score
    score_one_empty = 0

    # Check rows, columns, and blocks
    for i in range(N):
        # Check rows
        row_cells = get_row_cells(i)
        if empty_cells_in_region(row_cells) == 1:
            if occupied_neighbors(row_cells, opponent_occupied) and occupied_neighbors(row_cells, player_occupied):
                score_one_empty += 1

        # Check columns
        col_cells = get_col_cells(i)
        if empty_cells_in_region(col_cells) == 1:
            if occupied_neighbors(col_cells, opponent_occupied) and occupied_neighbors(col_cells, player_occupied):
                score_one_empty += 1

    # Check blocks
    for block_row in range(n):
        for block_col in range(m):
            block_cells = get_block_cells(block_row, block_col)
            if empty_cells_in_region(block_cells) == 1:
                if occupied_neighbors(block_cells, opponent_occupied) and occupied_neighbors(block_cells, player_occupied):
                    score_one_empty += 1

    # Return the score
    if node.current_player == my_player:
        return score_one_empty
    else:
        return -score_one_empty

def calc_score_center_moves(node):

    N = node.board.N  # Board size (N x N grid)
    center = (N - 1) / 2  # Center point (e.g., 4.5 for a 9x9 grid)

    def distance_to_center(row, col):
        row_weight = 0.66  # Weight for row distance (increase this to prioritize rows more)
        col_weight = 0.33 # Weight for column distance
        return ((row_weight * (row - center)) ** 2 + (col_weight * (col - center)) ** 2) ** 0.5

    def calculate_player_score(occupied_squares):

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





def score_partial_uninterrupted_lines(node):
    N = node.board.N  # Board size (N x N grid)
    if N<=4:
        return 0
    min_length = node.board.m + 1  # Minimum length of a sequence to be scored

    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )
    opponent_occupied = (
        node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1
    )
    column_weight = 0.25  # Weight factor for column lines

    def calculate_line_score(line, line_type, index):
        """
        Calculate the score for a given line by identifying uninterrupted sequences
        of the current player's cells that meet the minimum length requirement.
        Line scoring occurs only if there are no opponent squares near the start or end.
        """
        # Skip scoring if the line starts and ends at the edges based on its type
        if line_type == "row" and index == 0 and any(cell[0] == 0 for cell in line):
            return 0
        if line_type == "row" and index == N - 1 and any(cell[0] == N - 1 for cell in line):
            return 0
        if line_type == "col" and index == 0 and any(cell[1] == 0 for cell in line):
            return 0
        if line_type == "col" and index == N - 1 and any(cell[1] == N - 1 for cell in line):
            return 0

        # Calculate the score
        score = 0
        current_streak = 0

        for cell in line:
            if cell in current_player_occupied:
                current_streak += 1  # Increment streak if cell is occupied by the player
            else:
                # Score the streak if it meets the minimum length requirement
                if current_streak >= min_length:
                    score += compute_streak_score(current_streak, min_length)
                current_streak = 0

        # Add any remaining streak to the score if it meets the requirement
        if current_streak >= min_length:
            score += compute_streak_score(current_streak, min_length)

        # Apply column weight if the line is a column
        if line_type == "col":
            score *= column_weight
        return score

    def compute_streak_score(streak_length, min_length):
        """
        Compute the score for a streak based on its length relative to the minimum length.
        """
        diff = streak_length - min_length
        if diff == 0:
            return 0.5
        elif diff == 1:
            return 1
        elif diff == 2:
            return 2
        elif diff > 2:
            return 4
        return 0

    # Initialize total score
    total_score = 0

    # Check rows for uninterrupted sequences
    for row in range(N):
        row_cells = [(row, col) for col in range(N)]
        total_score += calculate_line_score(row_cells, "row", row)

    # Check columns for uninterrupted sequences
    for col in range(N):
        col_cells = [(row, col) for row in range(N)]
        total_score += calculate_line_score(col_cells, "col", col)

    return total_score


def calculate_score_difference(node):
    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]
def evaluate_node(node):
    N = node.board.N
    score_diff_game = calculate_score_difference(node)
    score_diff_center = calc_score_center_moves(node)

    current_player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    center = node.board.n -1
    if len(current_player_occupied) <= center:
        eval_func = score_diff_game + 2 * score_diff_center
        return eval_func

    score_one_empty = score_one_empty_in_region(node)
    score_mobility = calculate_mobility(node)
    score_block = score_block_occupation(node)
    score_line = score_partial_uninterrupted_lines(node)
    new = compute_blocking_advantage(node)

    eval_func = score_diff_game + score_one_empty + score_mobility + score_block + new + score_diff_center
    #print('score_diff: ', score_diff_game, 'new: ', new, 'score_center: ', score_diff_center,  'score_one_empty: ', score_one_empty , 'mobility: ',score_mobility, 'score block occupation: ',score_block, 'score line: ',score_line,'eval: ' ,eval_func)
    return eval_func

def compute_blocking_advantage(node):
    """
    Compute the blocking advantage for cells located only on the player's half of the board.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the blocking advantage.
    """
    N = node.board.N  # Board size (N x N grid)
    start_row = 0 if node.my_player == 1 else N-1
    # Determine the player's half of the board
    if node.my_player == 1:
        my_half = {(row, col) for row in range((N // 2) + 1) for col in range(N)}
    else:
        my_half = {(row, col) for row in range((N // 2), N) for col in range(N)}

    if node.my_player == node.current_player:  # If it is our turn
        player_playable_squares = node.player_squares()
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player  # Simulate opponent's turn
        opponent_playable_squares = simulated_state.player_squares()
    else:  # If it is not our turn
        opponent_playable_squares = node.player_squares()
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player  # Simulate our turn
        player_playable_squares = simulated_state.player_squares()

    # Compute the blocking advantage
    score = 0
    for row in range(N):
        for col in range(N):
            # Only consider cells on the player's half
            if (row, col) in my_half:
                # Check if the cell is empty, playable by the player, and not playable by the opponent
                if (node.board.get((row, col)) == 0 and
                        (row, col) in player_playable_squares and
                        (row, col) not in opponent_playable_squares):
                    score += 1
                    if row == start_row:
                        score+=0
    score = score / N#* ((len(node.occupied_squares1) + len(node.occupied_squares2)) / (N*N))
    return score


