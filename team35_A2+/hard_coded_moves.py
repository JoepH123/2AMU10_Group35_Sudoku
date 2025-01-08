from competitive_sudoku.sudoku import Move
from collections import deque

def get_heuristic_moves(node):
    """
    Determine the next move based on heuristic strategies.

    Parameters:
    - node: The current game state node.

    Returns:
    - A Move object if a valid move is found, else False.
    """
    m = node.board.m
    # Determine the squares occupied by the current player.
    my_player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)

    # If my player has fewer than m+1 occupied squares, compute start moves.
    if len(my_player_occupied) < m+1:
        return compute_start_moves(node)

    # Attempt to defend the corner if applicable.
    corner_defense = defend_corner(node)
    if corner_defense:
        return corner_defense

    # Attempt to block a area move.
    block_area = get_block_area_move(node)
    if block_area:
        return block_area

def defend_corner(node):
    """
    Check and defend a vulnerable corner (of size m x n) if the opponent poses a threat.
    Now generalized to depend on both m and n for risk squares and defensive stone placement.

    Parameters:
    - node: The current game state node.

    Returns:
    - A Move object if a defensive move is necessary, else False.
    """

    # The board is N x N, where N = m * n
    N = node.board.N
    m = node.board.m
    n = node.board.n

    if node.my_player == 1:
        # Check if top-left corner (rows < m-1, cols < n-1) is empty
        for row in range(m-1):
            for col in range(n-1):
                if node.board.get((row, col)) != 0:
                    return False

        # If opponent threatens squares below (row >= m), defend at row=m-1
        opponent_occupied = node.occupied_squares2
        player_occupied = node.occupied_squares1
        player_squares = node.player_squares()
        for col in range(n):
            for i in range(n-1):
                risk_sq = (m+i, col)
                if risk_sq in opponent_occupied:
                    for col2 in range(n-1):
                        defense_coordinates = (m - 1, col2)
                        if defense_coordinates not in player_occupied and defense_coordinates in player_squares:
                            return Move(defense_coordinates, node.solved_board_dict[defense_coordinates])

    elif node.my_player == 2:
        # Check if bottom-left corner (rows > N-m) is empty
        for row in range(N - m + 1, N):
            for col in range(n-1):
                if node.board.get((row, col)) != 0:
                    return False

        # If opponent threatens squares above (row <= N-(m+1)), defend at row=N-m
        opponent_occupied = node.occupied_squares1
        player_occupied = node.occupied_squares2
        player_squares = node.player_squares()
        for col in range(n):
            for i in range(n-1):
                risk_sq = (N - (m + 1 + i), col)
                if risk_sq in opponent_occupied:
                    for col2 in range(n-1):
                        defense_coordinates = (N - m, col2)
                        if defense_coordinates not in player_occupied and defense_coordinates in player_squares:
                            return Move(defense_coordinates, node.solved_board_dict[defense_coordinates])

    # If no threat is found or no corner defense is needed:
    return False


def compute_start_moves(node):
    """
    Compute the initial (opening) moves for a player based on their current state.
    We allow exactly m "opening moves" (vertically) for each player.

    Parameters:
    - node: The current game state node.

    Returns:
    - A Move object for the initial move, or False if no valid opening move is found.
    """
    N = node.board.N
    m = node.board.m
    n = node.board.n

    # Occupied squares for the current player
    player_occupied = (
        node.occupied_squares1
        if node.my_player == 1
        else node.occupied_squares2
    )

    if node.my_player == 1:
        # Look for an empty square in the top m rows, at column n-1 and return move on empty square
        if any(node.board.get((row, n-1)) == 0 for row in range(m)):
            for row in range(m):
                col = n-1
                coordinates = (row, col)
                if coordinates not in player_occupied:
                    return Move(coordinates, node.solved_board_dict[coordinates])
        else:
            return False

    elif node.my_player == 2:
        # Look for an empty square in the bottom m rows, at column n-1 and return move on empty square
        if any(node.board.get((row, n-1)) == 0 for row in range(N - m, N)):
            for row in range(N - 1, N - m - 1, -1):
                col = n-1
                coordinates = (row, col)
                if coordinates not in player_occupied:
                    return Move(coordinates, node.solved_board_dict[coordinates])
        else:
            return False

    return False


def bfs_paths(node, start_points, end_condition, corners_to_skip):
    """
    Perform a BFS from given starting points to find paths containing at most one empty cell.

    Parameters:
    ----------
    node : Node
        The current game state node containing board info and player squares.
    start_points : list of tuple
        The (row, col) coordinates from which the BFS will begin.
    end_condition : function
        A function that takes (row, col) and returns True if the path should be recorded.
    corners_to_skip : set of tuple
        (row, col) coordinates that must be excluded from any path.

    Returns:
    -------
    list of list of tuple
        A list of valid paths, where each path is a list of (row, col) steps.
        Each path can include at most one empty cell (i.e., board.get(...) == 0).
    """
    N = node.board.N
    occupied = node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    results = []
    visited = set()
    queue = deque([(r, c, [], 0) for (r, c) in start_points])  # (row, col, path, empty_count)

    while queue:
        row, col, path, empty_cnt = queue.popleft()
        if (row, col) in visited:
            continue
        visited.add((row, col))

        #extend path
        new_path = path + [(row, col)]
        new_empty_cnt = empty_cnt + (1 if node.board.get((row, col)) == 0 else 0)
        if new_empty_cnt > 1:
            continue

        # add path if conditions are met
        if end_condition(row, col):
            results.append(new_path)

        # Explore neighbors (prefer player-occupied cells over empties)
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        neighbors = [(r, c) for r, c in neighbors if 0 <= r < N and 0 <= c < N]
        neighbors.sort(key=lambda x: (node.board.get(x) == 0))  # Put empties last

        for nr, nc in neighbors:
            if (nr, nc) not in corners_to_skip:
                # Valid if either player's square or a free cell (and we haven't used up our free slot yet)
                if (node.board.get((nr, nc)) == 0 and new_empty_cnt < 1) or ((nr, nc) in occupied):
                    queue.append((nr, nc, new_path, new_empty_cnt))

    return results


def find_path_player_left(node):
    """
    Find paths (≤ 1 empty cell) starting from the left wall,
    ending at upper/right wall for P1 or lower/right wall for P2.
    """
    N = node.board.N
    occupied = node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2

    if node.my_player == 1:
        # Player 1: start from left wall (excluding row=0), end at top row or right wall
        corners = {(0, 0), (N-1, 0)}
        starts = [(r, 0) for r in range(1, N) if (r, 0) in occupied or node.board.get((r, 0)) == 0]
        end_cond = lambda r, c: (r == 0 and c != 0) or (c == N - 1)
    else:
        # Player 2: start from the left wall,
        # end at bottom row or right wall
        corners = {(0, N - 1), (N - 1, N - 1)}
        starts = [(r, 0) for r in range(N - 1) if (r, 0) in occupied or node.board.get((r, 0)) == 0]
        end_cond = lambda r, c: (r == N - 1 and c != 0) or (c == N - 1)

    paths = bfs_paths(node, starts, end_cond, corners)
    return filter_out_subsets(paths, N)


def find_path_player_right(node):
    """
    Find paths (≤ 1 empty cell) starting from the right wall,
    ending at top wall for P1 or bottom wall for P2.
    """
    N = node.board.N
    occupied = node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2

    if node.my_player == 1:
        # Player 1: start from right wall (excluding row=0), end at top row
        corners = {(0, N - 1), (N - 1, N - 1)}
        starts = [(r, N - 1) for r in range(1, N) if (r, N - 1) in occupied or node.board.get((r, N - 1)) == 0]
        end_cond = lambda r, c: (r == 0 and c != N - 1)
    else:
        # Player 2: start from right wall (excluding row=N-1), end at bottom row
        corners = {(0, 0), (N - 1, 0)}
        starts = [(r, N - 1) for r in range(N - 1) if (r, N - 1) in occupied or node.board.get((r, N - 1)) == 0]
        end_cond = lambda r, c: (r == N - 1 and c != N - 1)

    paths = bfs_paths(node, starts, end_cond, corners)
    return filter_out_subsets(paths, N)


def filter_out_subsets(paths, N):
    """
    Remove smaller paths that are strict subsets of larger ones.
    Restore any wall-adjacent cells removed in the process.
    """
    # Temporarily remove wall-adjacent cells (col == 0 or col == N-1)
    filtered, removed_coords = [], []
    for path in paths:
        keep, removed = [], []
        for (r, c) in path:
            if c in (0, N - 1):
                removed.append((r, c))
            else:
                keep.append((r, c))
        filtered.append(keep)
        removed_coords.append(removed)

    # Convert each filtered path to a set for subset checks
    set_paths = [set(path) for path in filtered]
    to_remove = set()

    # Mark any path that is a strict subset of another
    for i, s1 in enumerate(set_paths):
        for j, s2 in enumerate(set_paths):
            if i != j and s1 < s2:
                to_remove.add(i)

    # Only keep paths not marked for removal, then restore wall cells
    final_filtered = [(path, removed) for i, (path, removed) in enumerate(zip(filtered, removed_coords)) if i not in to_remove]
    return [path + removed for path, removed in final_filtered]


def is_valid_area_block(node, path):
    """
    Check if a path blocks an area under (P1) or above (P2) it
    by containing opponent squares or enough empty cells in that area.
    """
    N = node.board.N
    my_player = node.my_player
    opp_squares = node.occupied_squares2 if my_player == 1 else node.occupied_squares1

    # Specific small “invalid” path sets
    invalid_1 = {(1, 0), (1, 1), (0, 1)}
    invalid_2 = {(N - 2, 0), (N - 2, 1), (N - 1, 1)}
    if set(path) == invalid_1 or set(path) == invalid_2:
        return False

    # Path must have 1 empty cell
    if not any(node.board.get(p) == 0 for p in path):
        return False

    # Collect the "blocked" area (above for P2, below for P1)
    area = set()
    start_row_count, left_wall_count, right_wall_count = 0, 0, 0
    for (r, c) in path:
        # For Player 1, block upwards to row 0
        if my_player == 1:
            for rr in range(r, -1, -1):
                area.add((rr, c))
            if r == 0:
                start_row_count += 1

        # For Player 2, block downwards to row N-1
        else:
            for rr in range(r, N):
                area.add((rr, c))
            if r == N - 1:
                start_row_count += 1

        # Count how many path cells touch the left or right wall
        if c == 0:
            left_wall_count += 1
        if c == N - 1:
            right_wall_count += 1

    # Only 1 cell at start row/wall
    if start_row_count > 1 or left_wall_count > 1 or right_wall_count > 1:
        return False

    # No opponent squares in the blocked area
    if any(a in opp_squares for a in area):
        return False

    # At least 2 empty cells in that area to be considered a block
    return sum(node.board.get(a) == 0 for a in area) >= 2


def get_block_area_move(node):
    """
    Tries to find a valid path (left or right) that blocks the opponent's area.
    If found, returns a Move with one of the empty cells in that path.
    Otherwise returns False.
    """
    left_paths = find_path_player_left(node)
    right_paths = find_path_player_right(node)
    all_paths = left_paths + right_paths

    valid_paths = [p for p in all_paths if is_valid_area_block(node, p)]
    if valid_paths:
        longest = max(valid_paths, key=len)
        # Return the first empty cell in that path
        empty_cell = next((cell for cell in longest if node.board.get(cell) == 0), None)
        if empty_cell:
            return Move(empty_cell, node.solved_board_dict[empty_cell])
    return False







