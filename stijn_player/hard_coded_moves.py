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

    # If the current player has fewer than 2 occupied squares, compute start moves.
    if len(my_player_occupied) < m:
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
        # ----------------------------------------
        # 1) Check the top-left corner region:
        #    rows in [0..m-1], cols in [0..n-1].
        #    If ANY of these squares is NOT empty, no reason to defend.
        # ----------------------------------------
        for row in range(m):
            for col in range(n-1):
                if node.board.get((row, col)) != 0:
                    return False

        # ----------------------------------------
        # 2) Identify risk squares for Player 1:
        #    row = m, for all cols in [0..n-1].
        #    If the opponent occupies ANY of those squares,
        #    we place a defensive stone at (m-1, col).
        # ----------------------------------------
        opponent_occupied = node.occupied_squares2
        player_occupied = node.occupied_squares1
        player_squares = node.player_squares()
        for col in range(n):
            for i in range(n-1):
                risk_sq = (m+i, col)
                if risk_sq in opponent_occupied:
                    for col2 in range(n-1):
                        # Place defensive stone just one row up.
                        defense_coordinates = (m - 1, col2)
                        if defense_coordinates not in player_occupied and defense_coordinates in player_squares:
                            return Move(defense_coordinates, node.solved_board_dict[defense_coordinates])

    elif node.my_player == 2:
        # ----------------------------------------
        # 1) Check the bottom-left corner region:
        #    rows in [N-m..N-1], cols in [0..n-1].
        #    If ANY of these squares is NOT empty, no reason to defend.
        # ----------------------------------------
        for row in range(N - m, N):
            for col in range(n-1):
                if node.board.get((row, col)) != 0:
                    return False

        # ----------------------------------------
        # 2) Identify risk squares for Player 2:
        #    row = N - (m+1), for all cols in [0..n-1].
        #    If the opponent occupies ANY of those squares,
        #    we place a defensive stone at (N - m, col).
        # ----------------------------------------

        opponent_occupied = node.occupied_squares1
        player_occupied = node.occupied_squares2
        player_squares = node.player_squares()
        for col in range(n):
            for i in range(n-1):
                risk_sq = (N - (m + 1 + i), col)
                if risk_sq in opponent_occupied:
                    for col2 in range(n-1):
                        # Place defensive stone just one row closer to bottom corner.
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
    # N = total board dimension, where N = m*n
    # m, n = region sizes such that each region is m*n in size.
    # For demonstration, assume node has attributes node.m and node.n as well.
    N = node.board.N  # e.g., N = m*n
    m = node.board.m
    n = node.board.n   # Not strictly needed if we only require m.

    # Occupied squares for the current player
    player_occupied = (
        node.occupied_squares1
        if node.my_player == 1
        else node.occupied_squares2
    )

    # How many opening moves has the player already used?
    # This is simply the count of how many squares are occupied by that player.
    moves_used = len(player_occupied)

    # If we've already used up m moves, return False (no special opening move).
    if moves_used >= m:
        return False

    if node.my_player == 1:
        # Player 1: place stones from top to bottom
        # E.g., (0,1), (1,1), (2,1), ... up to (m-1,1)
        row = moves_used
        col = n-1
        coordinates = (row, col)
        return Move(coordinates, node.solved_board_dict[coordinates])

    elif node.my_player == 2:
        # Player 2: place stones from bottom to top
        # E.g., (N-1,1), (N-2,1), ... up to (N-m,1)
        row = N - 1 - moves_used
        col = n-1
        coordinates = (row, col)
        return Move(coordinates, node.solved_board_dict[coordinates])

    # If for some reason the player is neither 1 nor 2, return False.
    return False



from collections import deque

def bfs_paths(node, start_points, end_condition, corners_to_skip):
    """
    A generic BFS that returns all valid paths where each path has at most 1 empty cell.
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

        new_path = path + [(row, col)]
        new_empty_cnt = empty_cnt + (1 if node.board.get((row, col)) == 0 else 0)
        if new_empty_cnt > 1:
            continue

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
    Find paths (≤ 1 empty cell) starting from the left wall (or right wall, if player=2),
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
        # Player 2: start from the "other" side (the original code used col=0, row < N-1),
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
        corners = {(0, 0), (N - 1, 0)}  # Not used in the original code, but for symmetry
        starts = [(r, N - 1) for r in range(N - 1) if (r, N - 1) in occupied or node.board.get((r, N - 1)) == 0]
        end_cond = lambda r, c: (r == N - 1 and c != N - 1)

    paths = bfs_paths(node, starts, end_cond, corners)
    return filter_out_subsets(paths, N)


def filter_out_subsets(paths, N):
    """
    Remove smaller paths that are strict subsets of larger ones.
    Restore any wall-adjacent cells removed in the process.
    """
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

    set_paths = [set(p) for p in filtered]
    to_remove = set()
    for i, s1 in enumerate(set_paths):
        for j, s2 in enumerate(set_paths):
            if i != j and s1 < s2:
                to_remove.add(i)

    final_filtered = [(p, rm) for i, (p, rm) in enumerate(zip(filtered, removed_coords)) if i not in to_remove]
    return [p + rm for p, rm in final_filtered]


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

    # Path must have at least 1 empty cell
    if not any(node.board.get(p) == 0 for p in path):
        return False

    area = set()
    start_row_count, left_wall_count, right_wall_count = 0, 0, 0
    for (r, c) in path:
        if my_player == 1:
            for rr in range(r, -1, -1):
                area.add((rr, c))
            if r == 0:
                start_row_count += 1
        else:
            for rr in range(r, N):
                area.add((rr, c))
            if r == N - 1:
                start_row_count += 1

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







