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
    player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    opponent_occupied = (node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1)

    # Initialize score
    score_one_empty = 0

    # Check rows, columns, and blocks
    for i in range(N):
        # Check rows
        row_cells = get_row_cells(i)
        if empty_cells_in_region(row_cells) == 1:
            if node.current_player == node.my_player and occupied_neighbors(row_cells, opponent_occupied) and occupied_neighbors(row_cells, player_occupied):
                score_one_empty += 1
            if node.current_player != node.my_player and occupied_neighbors(row_cells, opponent_occupied):
                score_one_empty -= 1


        # Check columns
        col_cells = get_col_cells(i)
        if empty_cells_in_region(col_cells) == 1:
            if node.current_player == node.my_player and occupied_neighbors(col_cells, opponent_occupied) and occupied_neighbors(col_cells, player_occupied):
                score_one_empty += 1
            if node.current_player != node.my_player and occupied_neighbors(col_cells, opponent_occupied):
                score_one_empty -= 1

    # Check blocks
    for block_row in range(n):
        for block_col in range(m):
            block_cells = get_block_cells(block_row, block_col)
            if empty_cells_in_region(block_cells) == 1:
                if node.current_player == node.my_player and occupied_neighbors(block_cells, opponent_occupied) and occupied_neighbors(block_cells, player_occupied):
                    score_one_empty += 1
                if node.current_player != node.my_player and occupied_neighbors(block_cells, opponent_occupied):
                    score_one_empty -= 1
    return score_one_empty
def calc_score_center_moves(node):

    N = node.board.N  # Board size (N x N grid)
    center = (N+1) / 2  # Center point (e.g., 4.5 for a 9x9 grid)

    def distance_to_center(row, col):

        row_weight = 0.99  # Weight for row distance (increase this to prioritize rows more)
        col_weight = 0.01 # Weight for column distance
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



def calculate_score_difference(node):
    return node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]
def evaluate_node(node):
    N = node.board.N
    score_diff_game = calculate_score_difference(node)

    current_player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    center = node.board.m
    score_center = calc_score_center_moves(node)
    corner = compute_corner(node)
    if len(current_player_occupied) <= 3:
        eval_func = score_diff_game + 2 * score_center + corner
        return eval_func
    # if center < len(current_player_occupied) <= center+3:
    #     score_hor = score_horizontal_wall_moves_with_adjacent(node)
    #     eval_func = score_diff_game + score_hor
    #     return eval_func

    score_one_empty = score_one_empty_in_region(node)
    score_mobility = calculate_mobility(node)
    score_block = score_block_occupation(node)
    new = compute_blocking_advantage(node)
    wall = score_wall_and_adjacent_moves(node)
    corner = compute_corner(node)
    zero = punish_zero(node)


    eval_func = score_diff_game + score_mobility + score_one_empty  + zero + score_center
    #print('score_diff: ', score_diff_game, 'score_center: ', score_center,  'score_one_empty: ', score_one_empty , 'mobility: ',score_mobility, "new: ", 2*new, 'eval: ' ,eval_func)


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
        my_half = {(row, col) for row in range(node.board.m) for col in range(N)}
    else:
        my_half = {(row, col) for row in range(N - node.board.m, N) for col in range(N)}


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
                if (node.board.get((row, col)) == 0 and (row, col) in opponent_playable_squares): #(row, col) in player_playable_squares and

                    score -= 1
                if (node.board.get((row, col)) == 0 and (row, col) in player_playable_squares):

                    score += 1
                    if row == start_row:
                        score+=0
    score = score / N#* ((len(node.occupied_squares1) + len(node.occupied_squares2)) / (N*N))
    return score



def score_wall_and_adjacent_moves(node):
    """
    Calculate bonus points for occupied squares that:
    - Have own moves both to the left and right, or
    - Are adjacent to a wall and have an own move on the other side.
    - Are not in the first or last row.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the bonus points.
    """
    N = node.board.N  # Board size (N x N grid)
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    def is_left_and_right_own(row, col):
        """
        Check if there are own moves both to the left and right of the square.
        """
        left = (row, col - 1)
        right = (row, col + 1)

        return (
                (col > 0 and left in current_player_occupied) and
                (col < N - 1 and right in current_player_occupied)
        )

    def is_wall_left_or_right_and_own_adjacent(row, col):
        """
        Check if the square has a wall on one side and an own move on the other side.
        """
        left_wall = col == 0
        right_wall = col == N - 1
        left_own = (row, col - 1) in current_player_occupied if col > 0 else False
        right_own = (row, col + 1) in current_player_occupied if col < N - 1 else False

        return (left_wall and right_own) or (right_wall and left_own)

    # Calculate the score
    score = 0
    for row in range(1, N - 1):  # Exclude the first and last rows
        for col in range(N):  # Check all columns
            if (row, col) in current_player_occupied:  # Check only occupied squares
                if is_left_and_right_own(row, col) or is_wall_left_or_right_and_own_adjacent(row, col):
                    score += 1

    return score

def compute_corner(node):
    """
    Compute the blocking advantage for cells located only on the player's half of the board.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the blocking advantage.
    """

    if node.my_player ==1:
        # Define players' occupied squares
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        opponent_occupied = (node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1)
        score = 0

        corner = [(0,1), (1,1), (1,0)]
        for cell in corner:
            if cell in player_occupied:
                score+=10
            if cell in opponent_occupied:
                return 0

        if (0,0) in player_occupied:
            score = -100
        return score
    if node.my_player ==2:
        N = node.board.N
        # Define players' occupied squares
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        opponent_occupied = (node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1)
        score = 0

        corner = [(N-2,0), (N-2,1), (N-1,1)]
        for cell in corner:
            if cell in player_occupied:
                score+=10
            if cell in opponent_occupied:
                return 0

        if (N-1,0) in player_occupied:
            score = -100
        return score

def punish_zero(node):
    if node.my_player ==1:
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        score=0
        if (0,0) in player_occupied:
            score = -100
        return score
    if node.my_player ==2:
        N = node.board.N
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        score=0
        if (N-1,0) in player_occupied:
            score = -100
        return score
