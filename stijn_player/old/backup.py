def score_block_occupation(self, node):
    """
    Evaluate the game state with the following rules:
    - +1 for each block where the current player has at least one occupied square.
    - +2 additional score for each block adjacent to the opponent's starting row.
    - Exclude blocks adjacent to the player's starting row.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the evaluation.
    """
    N = node.board.N  # Board size (N x N grid)
    n, m = node.board.n, node.board.m  # Block dimensions

    # Determine the starting rows for both players
    my_starting_row = 0 if node.my_player == 1 else N - 1
    opponent_starting_row = N - 1 if node.my_player == 1 else 0

    # Identify rows adjacent to the player's and opponent's starting rows
    adjacent_to_my_rows = {my_starting_row}
    if my_starting_row - 1 >= 0:
        adjacent_to_my_rows.add(my_starting_row - 1)
    if my_starting_row + 1 < N:
        adjacent_to_my_rows.add(my_starting_row + 1)

    adjacent_to_opponent_rows = {opponent_starting_row}
    if opponent_starting_row - 1 >= 0:
        adjacent_to_opponent_rows.add(opponent_starting_row - 1)
    if opponent_starting_row + 1 < N:
        adjacent_to_opponent_rows.add(opponent_starting_row + 1)

    # Identify blocks adjacent to the player's and opponent's starting rows
    adjacent_to_my_blocks = set()
    for r in adjacent_to_my_rows:
        block_row = r // m
        for block_col in range(n):
            adjacent_to_my_blocks.add((block_row, block_col))

    adjacent_to_opponent_blocks = set()
    for r in adjacent_to_opponent_rows:
        block_row = r // m
        for block_col in range(n):
            adjacent_to_opponent_blocks.add((block_row, block_col))

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
            # Skip blocks adjacent to the player's starting row
            if (block_row, block_col) in adjacent_to_my_blocks:
                continue

            block_cells = get_block_cells(block_row, block_col)

            # Check if the current player occupies at least one square in this block
            if any(cell in current_player_occupied for cell in block_cells):
                # +1 for occupying any block
                score += 1

                # +2 bonus if the block is adjacent to the opponent's starting row
                # if (block_row, block_col) in adjacent_to_opponent_blocks:
                #     score += 2

    return score


def calculate_mobility(self, node):
    N = node.board.N
    if node.my_player == node.current_player: # if it is our turn
        return len(node.player_squares()) / N

    else: # if it is not our turn
        simulated_state = copy.deepcopy(node)
        simulated_state.current_player = 3 - node.current_player # make a copy of the game state with our player to move
        return len(simulated_state.player_squares()) / N
def score_one_empty_in_region(self, node):

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

def calc_score_center_moves(self, node):

    N = node.board.N  # Board size (N x N grid)
    center = (N - 1) / 2  # Center point (e.g., 4.5 for a 9x9 grid)

    def distance_to_center(row, col):
        row_weight = 0.666  # Weight for row distance (increase this to prioritize rows more)
        col_weight = 0.333  # Weight for column distance
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

def evaluate_occupation_in_half(self, node):

    N = node.board.N  # Board size (N x N grid)
    my_player = node.my_player
    opponent = 3 - my_player  # The other player (1 or 2)

    # Define halves based on whether N is even or odd
    if N % 2 == 0:  # Even grid
        midpoint = N // 2
        if my_player == 1:
            my_half = set((row, col) for row in range(midpoint + 1) for col in range(N))
            opponent_half = set((row, col) for row in range(midpoint, N) for col in range(N))

        else:
            my_half = set((row, col) for row in range(midpoint - 1, N) for col in range(N))
            opponent_half = set((row, col) for row in range(midpoint) for col in range(N))

    else:  # Uneven grid
        midpoint = N // 2
        if my_player == 1:
            my_half = set((row, col) for row in range(midpoint + 2) for col in range(N))
            opponent_half = set((row, col) for row in range(midpoint + 1, N) for col in range(N))

        else:
            my_half = set((row, col) for row in range(midpoint - 2, N) for col in range(N))
            opponent_half = set((row, col) for row in range(midpoint) for col in range(N))

    # Get the occupied squares for both players
    my_occupied = (
        node.occupied_squares1 if my_player == 1 else node.occupied_squares2
    )
    opponent_occupied = (
        node.occupied_squares2 if my_player == 1 else node.occupied_squares1
    )

    # Calculate rewards and penalties
    reward = sum(1 for square in my_occupied if square in opponent_half)
    penalty = sum(1 for square in opponent_occupied if square in my_half)

    return reward, -penalty

def evaluate_balance(self, node):

    N = node.board.N  # Board size (N x N grid)
    half = N // 2  # Midpoint of the board

    # Define players' occupied squares
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    # Count occupied cells in left and right halves
    left_count = sum(1 for r, c in current_player_occupied if c < half)
    right_count = sum(1 for r, c in current_player_occupied if c >= half)

    # Calculate balance score
    balance = 1 - abs(left_count - right_count) / max(1, (left_count + right_count))

    return balance

def score_enclosed_cells(self, node):
    """
    Evaluate the game state by scoring enclosed empty cells. A cell is enclosed
    if all 8 directions (including diagonals) are surrounded by the current player's
    occupied cells or the board boundary. Score is only awarded if the enclosed
    region contains more than 1 empty cell.

    Parameters:
    - node: The GameState object.

    Returns:
    - An integer score representing the evaluation.
    """
    N = node.board.N  # Board size (N x N grid)
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )
    opponent_occupied = (
        node.occupied_squares2 if node.my_player == 1 else node.occupied_squares1
    )

    # Track visited cells to avoid re-checking
    visited = set()

    def is_valid_cell(row, col):
        """
        Check if a cell is within the board and not already visited.
        """
        return 0 <= row < N and 0 <= col < N and (row, col) not in visited

    def get_neighbors(row, col):
        """
        Get all 8 neighboring cells (including diagonals).
        """
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal directions
        ]
        return [(row + dr, col + dc) for dr, dc in directions if 0 <= row + dr < N and 0 <= col + dc < N]

    def is_enclosed(empty_cells):
        """
        Check if all cells in the region are surrounded by the current player's cells or walls.
        """
        for r, c in empty_cells:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in current_player_occupied and (nr, nc) not in empty_cells:
                    return False
        return True

    def explore_region(start_row, start_col):
        """
        Explore a region starting from an empty cell and return all connected empty cells.
        """
        stack = [(start_row, start_col)]
        region = set()
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or (r, c) in current_player_occupied or (r, c) in opponent_occupied:
                continue
            visited.add((r, c))
            region.add((r, c))
            for neighbor in get_neighbors(r, c):
                if neighbor not in visited:
                    stack.append(neighbor)
        return region

    # Initialize score
    score = 0

    # Iterate over the board to find enclosed regions
    for row in range(N):
        for col in range(N):
            if (row, col) not in visited and node.board.get((row, col)) == 0:
                # Explore the region starting from this empty cell
                region = explore_region(row, col)

                # Only consider regions with more than 1 empty cell
                if len(region) > 1 and is_enclosed(region):
                    # Score: +1 for each empty cell in the region
                    score += len(region)

    return score

def score_partial_uninterrupted_lines(self, node):

    N = node.board.N  # Board size (N x N grid)
    min_length = node.board.N // 2  # Minimum length of a sequence to be scored
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    def calculate_line_score(line):
        """
        Calculate the score for a given line by identifying uninterrupted sequences
        of the current player's cells that meet the minimum length requirement.
        """
        score = 0
        current_streak = 0

        for cell in line:
            if cell in current_player_occupied:
                current_streak += 1  # Increment streak if cell is occupied by the player
            else:
                # Score the streak if it meets the minimum length requirement
                if current_streak >= min_length:
                    score += current_streak
                current_streak = 0

        # Add any remaining streak to the score if it meets the requirement
        if current_streak >= min_length:
            score += current_streak

        return score

    # Initialize total score
    total_score = 0

    # Check rows for uninterrupted sequences
    for row in range(N):
        row_cells = [(row, col) for col in range(N)]
        total_score += calculate_line_score(row_cells)

    # Check columns for uninterrupted sequences
    for col in range(N):
        col_cells = [(row, col) for row in range(N)]
        total_score += calculate_line_score(col_cells)

    return total_score


def evaluate(self, node):
    N = node.board.N
    score_diff_game = node.scores[node.my_player - 1] - node.scores[1-(node.my_player - 1)]
    score_diff_center = self.calc_score_center_moves(node)
    reward_occupation, penalty_occupation = self.evaluate_occupation_in_half(node)
    score_one_empty = self.score_one_empty_in_region(node)
    mobility = self.calculate_mobility(node)
    block_occupation = self.score_block_occupation(node)
    close = self.score_enclosed_cells(node)
    line = self.score_partial_uninterrupted_lines(node)
    if N > 4:
        score_balance = self.evaluate_balance(node)
    else:
        score_balance = 0
    #eval_func = score_diff_game + 2*score_diff_center + 0.25*reward_occupation + 0.25*penalty_occupation + score_one_empty + 2*score_balance
    eval_func = score_diff_game + 0.5*mobility + 0.5*score_diff_center + score_one_empty + 0.5*block_occupation + 0.5*line
    #print('score_diff: ', score_diff_game, 'score_center: ', score_diff_center,  'score_one_empty: ', score_one_empty , 'mobility: ',0.5* mobility, 'reward occupation: ',0.5*block_occupation,'eval: ' ,eval_func)
    return eval_func