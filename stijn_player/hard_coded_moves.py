from competitive_sudoku.sudoku import Move

def opening_game_phase_moves(node):
    # if node.board.N <= 4:
    #     return False
    current_player_occupied = (node.occupied_squares1 if node.my_player == 1 else node.occupied_squares2)
    if len(current_player_occupied) < 2:
        return compute_start_moves(node)
    return defend_corner(node)

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
