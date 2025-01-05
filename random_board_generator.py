import random

def generate_board():
    """
    Generates a 3x3 game board with a single randomly placed value in the top and bottom row.
    
    Constraints:
    - The board consists of 9 rows and 9 columns, initialized with '.'
    - A random integer between 1 and 9 is placed in the top row with a '+' sign.
    - A different random integer between 1 and 9 is placed in the bottom row with a '-' sign.
    - The first three columns (indices 0 to 2) cannot be selected for placing values.
    - Given that both values are placed in the same column, the top and bottom values must differ. This way the sudoku remains solvable. 
    - The generated board is saved to 'boards/board-3x3-random_start.txt'.

    Outputs:
    - A text file containing the board layout, moves, taboo moves, and scores.
    """
    # Board initialization
    rows = 3
    columns = 3
    board = [['.'] * 9 for _ in range(9)]
    moves = []
    taboo_moves = []
    scores = [0, 0]

    # Randomly select columns (between 3 and 8) for the top and bottom rows
    top_column = random.randint(3, 8)
    bottom_column = random.randint(3, 8)

    # Randomly select integers between 1 and 9
    top_value = random.randint(1, 9)
    bottom_value = random.randint(1, 9)

    # Make sure that if both starting numbers are in the same column, the values are not the same. This way the sudoku remains solvable. 
    if top_column == bottom_column:
        while top_value == bottom_value:
            bottom_value = random.randint(1, 9)

    # Place the values on the board
    board[0][top_column] = f"{top_value}+"
    board[8][bottom_column] = f"{bottom_value}-"

    # Track moves
    moves.append((0, top_column, top_value))
    moves.append((8, bottom_column, bottom_value))

    # Write the result to a file
    with open('boards/board-3x3-random_start.txt', 'w') as file:
        file.write(f"rows = {rows}\n")
        file.write(f"columns = {columns}\n")
        file.write("board =\n")
        for row in board:
            file.write("   " + "  ".join(row) + "\n")
        move_strings = [f"({m[0]},{m[1]}) -> {m[2]}" for m in moves]
        file.write(f"moves = [{', '.join(move_strings)}]\n")
        file.write(f"taboo-moves = {taboo_moves}\n")
        file.write(f"scores = {scores}\n")

if __name__ == "__main__":
    generate_board()
