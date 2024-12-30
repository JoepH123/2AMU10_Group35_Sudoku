import numpy as np
import random
from env import DQLGameState

def print_board(board):
    """Utility function to print the board in a readable format."""
    symbol_map = {1: 'X', -1: 'O', 0: '.'}
    # Print column headers
    print("   " + " ".join(str(c) for c in range(9)))
    print("  +" + "--"*9 + "+")
    for r in range(9):
        row_symbols = [symbol_map[cell] for cell in board[r]]
        print(f"{r} |" + " ".join(row_symbols) + "|")
    print("  +" + "--"*9 + "+")

def play_game(state):
    """Function to play a single game."""
    print("Initial Board:")
    print_board(state.board)
    print("-" * 40)

    move_number = 1

    while True:
        current_player = state.current_player
        player_symbol = 'Player 1 (X)' if current_player == 1 else 'Player 2 (O)'

        # Get all possible moves for the current player
        moves = state.get_all_moves()

        if not moves:
            # Current player cannot move
            print(f"{player_symbol} cannot make a move.")

            # Check if the other player can still move
            other_player = -current_player
            other_moves = state.get_all_moves(player=other_player)

            if not other_moves:
                # Neither player can move; game over
                print("Both players cannot make any more moves. Game Over!")
                break
            else:
                # Switch to the other player
                print(f"{player_symbol} passes the turn. Switching to the other player.")
                state.current_player = other_player
                print("-" * 40)
                continue

        # Choose a random move
        action = random.choice(moves)

        # Perform the move
        reward, done, info = state.step(action)

        # Determine which player just made the move
        last_player = -state.current_player  # Since step() switches the player

        # Print move details
        print(f"Move {move_number}: { 'Player 1 (X)' if last_player == 1 else 'Player 2 (O)' } placed on {action}, Reward: {reward}")
        print("Current Score:", info["score"])
        print_board(state.board)
        print("-" * 40)

        move_number += 1

        if done:
            print("Game Over!")
            break

    # Final Scores
    print("Final Scores:")
    print(f"Player 1 (X): {state.score[0]} points")
    print(f"Player 2 (O): {state.score[1]} points")

    # Determine the winner
    if state.score[0] > state.score[1]:
        print("Player 1 (X) wins!")
    elif state.score[1] > state.score[0]:
        print("Player 2 (O) wins!")
    else:
        print("It's a tie!")

    # Final Board
    print("Final Board:")
    print_board(state.board)

def main():
    state = DQLGameState()

    while True:
        play_game(state)

        # Ask if the user wants to play again
        play_again = input("Do you want to play again? (y/n): ").strip().lower()
        if play_again == 'y':
            state.reset()
        else:
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()
