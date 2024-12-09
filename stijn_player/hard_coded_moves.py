from competitive_sudoku.sudoku import Move
import time
from collections import deque
def get_heuristic_moves(node):
    # if node.board.N <= 4:
    #     return False
    current_player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    if len(current_player_occupied) < 2:
        return compute_start_moves(node)
    corner_defense = defend_corner(node)
    if corner_defense:
        return corner_defense
    block_region = get_block_region_move(node)
    if block_region:
        return block_region

def defend_corner(node):
    N = node.board.N
    if node.my_player == 1:
        if not (node.board.get((0, 0)) == 0 and node.board.get((1, 0)) == 0):
            return False
        opponent_occupied = node.occupied_squares2
        risk_squares = [(2,1), (2,0)]
        for sq in risk_squares:
            if sq in opponent_occupied:
                coordinates = (1,0)
                return Move(coordinates, node.solved_board_dict[coordinates])

    elif node.my_player == 2:
        if not (node.board.get((N-1, 0)) == 0 and node.board.get((N-2, 0)) == 0):
            return False
        opponent_occupied = node.occupied_squares1
        risk_squares = [(N-3,1), (N-3,0)]
        for sq in risk_squares:
            if sq in opponent_occupied:
                coordinates = (N-2,0)
                return Move(coordinates, node.solved_board_dict[coordinates])
    return False

def compute_start_moves(node):
    N = node.board.N
    if node.my_player == 1:
        # Define players' occupied squares
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        if len(player_occupied)==0:
            coordinates = (0,1)
            return Move(coordinates, node.solved_board_dict[coordinates])

        if len(player_occupied)==1:
            coordinates = (1,1)
            return Move(coordinates, node.solved_board_dict[coordinates])


    elif node.my_player == 2:
        # Define players' occupied squares
        player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
        if len(player_occupied)==0:
            coordinates = (N-1,1)
            return Move(coordinates, node.solved_board_dict[coordinates])
        if len(player_occupied)==1:
            coordinates = (N-2,1)
            return Move(coordinates, node.solved_board_dict[coordinates])

    return False


def find_path_player_left(node):
    """
    Find a path that minimizes the number of empty cells and ensures at most one empty cell.
    The path can start at the left wall (excluding row 0 and (0, 0)) if Player 1, or at the right wall if Player 2.
    The path can end at the upper wall, the right wall, or the left wall based on the player's position.

    Parameters:
    - node: The GameState object.

    Returns:
    - A list of tuples representing the empty cells on the path.
    """
    N = node.board.N  # Board size (N x N grid)
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    def is_valid_cell(row, col, visited, empty_cell_count):
        """
        Check if a cell is valid for inclusion in the path.
        - It must be within bounds.
        - It must not be a restricted starting cell like (0, 0) or (N-1, N-1).
        - It must be empty or occupied by the current player.
        - It must not have been visited before.
        - It must not exceed the maximum empty cell limit.
        """
        return (
                (row, col) != (0, 0) and (row, col) != (N - 1, 0) and
                0 <= row < N and 0 <= col < N and
                (row, col) not in visited and
                (
                        (node.board.get((row, col)) == 0 and empty_cell_count < 1) or
                        (row, col) in current_player_occupied
                )
        )

    def bfs():
        """
        Perform BFS to find a path based on the current player's perspective.
        """
        # Determine starting and ending walls based on the player
        if node.my_player == 1:
            # Player 1 starts from the left wall (excluding row 0)
            starting_points = [
                (row, 0, [], 0, (row, 0))  # (row, col, path, empty_cell_count)
                for row in range(1, N)
                if node.board.get((row, 0)) == 0 or (row, 0) in current_player_occupied
            ]
            ending_conditions = lambda r, c: (r == 0 and c != 0) or (c == N - 1)  # Ends at upper or right wall
        else:
            # Player 2 starts from the right wall (excluding row N-1)
            starting_points = [
                (row, 0, [], 0, (row, 0))  # (row, col, path, empty_cell_count)
                for row in range(N - 1)
                if node.board.get((row, 0)) == 0 or (row, 0) in current_player_occupied
            ]
            ending_conditions = lambda r, c: (r == N-1 and c != 0) or (c == N-1)  # Ends at upper or left wall

        # Initialize the queue
        queue = deque(starting_points)
        visited = set()
        best_path = []
        min_empty_cells = float('inf')
        all_valid_paths = []
        while queue:
            row, col, path, empty_cell_count, origin = queue.popleft()

            if (row, col, origin) in visited:
                continue
            visited.add((row, col, origin))

            # Add the current cell to the path
            new_path = path + [(row, col)]
            new_empty_cell_count = empty_cell_count + (1 if node.board.get((row, col)) == 0 else 0)

            # Stop exploring if empty cell count exceeds the limit
            if new_empty_cell_count > 1:
                continue

            # Check if the path satisfies the ending conditions
            if ending_conditions(row, col) and new_empty_cell_count <= 1:
                all_valid_paths.append(new_path)

            # Explore neighbors (horizontal and vertical only)
            neighbors = [
                (row - 1, col), (row + 1, col),  # Vertical
                (row, col - 1), (row, col + 1)   # Horizontal
            ]

            # Filter neighbors to include only valid board coordinates
            neighbors = [
                (nr, nc) for nr, nc in neighbors
                if 0 <= nr < N and 0 <= nc < N  # Ensure within bounds
            ]

            neighbors.sort(key=lambda x: node.board.get(x) == 0)  # Prefer player-occupied cells

            for nr, nc in neighbors:
                if is_valid_cell(nr, nc, visited, new_empty_cell_count):
                    queue.append((nr, nc, new_path, new_empty_cell_count, origin))

        return all_valid_paths

    # Perform the BFS to find the best path
    paths = bfs()
    paths = filter_out_subsets(paths, N)
    # Extract and return the empty cells on the best path
    return paths#[cell for path in paths for cell in path if node.board.get(cell) == 0]



def find_path_player_right(node):
    """
    Find a path that minimizes the number of empty cells and ensures at most one empty cell.
    The path can start at the left wall (excluding row 0 and (0, 0)) if Player 1, or at the right wall if Player 2.
    The path can end at the upper wall, the right wall, or the left wall based on the player's position.

    Parameters:
    - node: The GameState object.

    Returns:
    - A list of tuples representing the empty cells on the path.
    """
    N = node.board.N  # Board size (N x N grid)
    current_player_occupied = (
        node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2
    )

    def is_valid_cell(row, col, visited, empty_cell_count):
        """
        Check if a cell is valid for inclusion in the path.
        - It must be within bounds.
        - It must not be a restricted starting cell like (0, 0) or (N-1, N-1).
        - It must be empty or occupied by the current player.
        - It must not have been visited before.
        - It must not exceed the maximum empty cell limit.
        """
        return (
                (row, col) != (0, N-1) and (row, col) != (N - 1, N-1) and
                0 <= row < N and 0 <= col < N and
                (row, col) not in visited and
                (
                        (node.board.get((row, col)) == 0 and empty_cell_count < 1) or
                        (row, col) in current_player_occupied
                )
        )

    def bfs():
        """
        Perform BFS to find a path based on the current player's perspective.
        """
        # Determine starting and ending walls based on the player
        if node.my_player == 1:
            # Player 1 starts from the left wall (excluding row 0)
            starting_points = [
                (row, N-1, [], 0, (row, N-1))  # (row, col, path, empty_cell_count)
                for row in range(1, N)
                if node.board.get((row, N-1)) == 0 or (row, N-1) in current_player_occupied
            ]
            ending_conditions = lambda r, c: (r == 0 and c != N-1)   # Ends at upper or right wall
        else:
            # Player 2 starts from the right wall (excluding row N-1)
            starting_points = [
                (row, N-1, [], 0, (row, N-1))  # (row, col, path, empty_cell_count)
                for row in range(N - 1)
                if node.board.get((row, N-1)) == 0 or (row, N-1) in current_player_occupied
            ]
            ending_conditions = lambda r, c: (r == N-1 and c != N-1)  # Ends at upper or left wall

        # Initialize the queue
        queue = deque(starting_points)
        visited = set()
        best_path = []
        min_empty_cells = float('inf')
        all_valid_paths = []
        while queue:
            row, col, path, empty_cell_count, origin = queue.popleft()

            if (row, col, origin) in visited:
                continue
            visited.add((row, col, origin))

            # Add the current cell to the path
            new_path = path + [(row, col)]
            #print(new_path)
            new_empty_cell_count = empty_cell_count + (1 if node.board.get((row, col)) == 0 else 0)

            # Stop exploring if empty cell count exceeds the limit
            if new_empty_cell_count > 1:
                continue

            # Check if the path satisfies the ending conditions
            if ending_conditions(row, col) and new_empty_cell_count <= 1:
                all_valid_paths.append(new_path)

            # Explore neighbors (horizontal and vertical only)
            neighbors = [
                (row - 1, col), (row + 1, col),  # Vertical
                (row, col - 1), (row, col + 1)   # Horizontal
            ]

            # Filter neighbors to include only valid board coordinates
            neighbors = [
                (nr, nc) for nr, nc in neighbors
                if 0 <= nr < N and 0 <= nc < N  # Ensure within bounds
            ]

            neighbors.sort(key=lambda x: node.board.get(x) == 0)  # Prefer player-occupied cells

            for nr, nc in neighbors:
                if is_valid_cell(nr, nc, visited, new_empty_cell_count):
                    queue.append((nr, nc, new_path, new_empty_cell_count, origin))

        return all_valid_paths

    # Perform the BFS to find the best path
    paths = bfs()
    paths = filter_out_subsets(paths, N)
    # Extract and return the empty cells on the best path
    return paths#[cell for path in paths for cell in path if node.board.get(cell) == 0]


def is_valid_region_block(node, path):
    """
    Check if there are opponent squares in the region under a given path,
    if the region has 2 or fewer empty cells, and if there is only 1 cell in the start row.
    The path can have multiple turns, and the region is dynamically calculated.

    Parameters:
    - node: The GameState object.
    - path: A list of tuples representing the cells in the path.

    Returns:
    - True if there are opponent squares in the region or more than 2 empty cells,
      False otherwise.
    """
    N = node.board.N  # Board size (N x N grid)
    current_player = node.my_player
    opponent_squares = (
        node.occupied_squares2 if current_player == 1 else node.occupied_squares1
    )
    invalid_path_set = {(1, 0), (1, 1), (0, 1)}
    if set(path) == invalid_path_set:
        return False

    if len([cell for cell in path if node.board.get(cell) == 0])==0:
        return False
    # Dynamically compute the region under the path
    region = set()
    start_row_count = 0
    left_wall_count = 0
    right_wall_count = 0# Count cells in the start row
    for i, (row, col) in enumerate(path):
        # Add all rows below (for Player 1) or above (for Player 2)
        if current_player == 1:
            for r in range(row, -1, -1):
                region.add((r, col))
            if row == 0:  # Player 1's start row
                start_row_count += 1
            if col==0:
                left_wall_count +=1
            if col == N-1:
                right_wall_count +=1
        else:
            for r in range(row, N):
                region.add((r, col))
            if row == N - 1:  # Player 2's start row
                start_row_count += 1
            if col==0:
                left_wall_count +=1
            if col == N-1:
                right_wall_count +=1

    # Check if only 1 cell in the start row
    if start_row_count > 1:
        return False
    if left_wall_count > 1:
        return False
    if right_wall_count > 1:
        return False
        # Count empty cells and check for opponent squares
    empty_cells = 0
    for cell in region:
        if cell in opponent_squares:
            return False  # Opponent square found in the region

    for cell in region:
        if node.board.get(cell) == 0:
            empty_cells += 1
            if empty_cells >= 2:  # Early exit if more than 2 empty cells
                return True

    # If no opponent squares and 2 or more empty cells, return False
    return False

def filter_out_subsets(paths, N):
    filtered_lists = []
    removed_coords = []

    for coord_list in paths:
        new_list = []
        removed = []
        for (r, c) in coord_list:
            if c == 0 or c == N-1:
                removed.append((r, c))
            else:
                new_list.append((r,c))
        filtered_lists.append(new_list)
        removed_coords.append(removed)

    # Step 2: Remove subset lists
    set_lists = [set(lst) for lst in filtered_lists]

    to_remove = set()
    for i, s1 in enumerate(set_lists):
        for j, s2 in enumerate(set_lists):
            if i != j and s1 < s2:  # s1 is strictly contained in s2
                to_remove.add(i)

    final_filtered_lists = [lst for i, lst in enumerate(filtered_lists) if i not in to_remove]
    final_removed_coords = [rm for i, rm in enumerate(removed_coords) if i not in to_remove]

    # Step 3: Reintroduce original boundary coordinates
    restored_lists = []
    for coords, removed in zip(final_filtered_lists, final_removed_coords):
        restored_list = coords + removed
        restored_lists.append(restored_list)
    return restored_lists

def get_block_region_move(node):
    t = time.time()
    left_path = find_path_player_left(node)
    right_path = find_path_player_right(node)
    paths = left_path + right_path
    valid = [is_valid_region_block(node, path) for path in paths]
    result = [path for path, flag in zip(paths, valid) if flag]
    #print(result)
    if len(result)>0:
        longest_path = max(result, key=len)
        coordinates = [cell for cell in longest_path if node.board.get(cell) == 0][0]
        return Move(coordinates, node.solved_board_dict[coordinates])
    else:
        return False







