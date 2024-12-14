# env.py

import numpy as np

class DQLGameState:
    def __init__(self):
        self.reset()  # Reset for new game

    def reset(self):
        """Reset the game state to its initial state."""
        self.board = np.zeros((9, 9), dtype=int)  # Reset the board to an empty state
        self.current_player = 1  # Player 1 starts
        self.score = (0, 0)  # Reset scores for both players
        self.first_move_done = {1: False, -1: False}  # Track if each player has made their first move
        self.player_positions = {1: [], -1: []}  # Clear player positions
        self.player_can_move = {1: True, -1: True}  # Reset move availability for both players

    def get_all_moves(self, player=None):
        """Return a list of all possible moves for the specified player.
           If player is None, use the current player.
           A move is a tuple (row, col)."""
        if player is None:
            player = self.current_player

        # If the player cannot move, return an empty list
        if not self.player_can_move[player]:
            return []

        moves = []
        if not self.first_move_done[player]:
            # First move constraints:
            # Player 1 must choose from top row (row = 0)
            # Player -1 must choose from bottom row (row = 8)
            start_row = 0 if player == 1 else 8
            for c in range(9):
                if self.board[start_row, c] == 0:
                    moves.append((start_row, c))
            return moves

        # If first move is done, must choose a cell adjacent to any previously placed cell by this player
        occupied_positions = self.player_positions[player]
        if not occupied_positions:
            # No occupied positions, should not happen if first_move_done is True
            return []

        possible_moves = set()

        for (r, c) in occupied_positions:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        if self.board[nr, nc] == 0:
                            possible_moves.add((nr, nc))

        moves = list(possible_moves)
        return moves

    def is_terminal(self):
        """Check if the game is over.
           The game is over if both players cannot make any moves."""
        return not (self.player_can_move[1] or self.player_can_move[-1])

    def step(self, action):
        """Perform the given action (move) for the current player.
           action is (row, col).

           Returns:
             reward (float): The points gained by making this move
             done (bool): True if the game ended after this move
             info (dict): Additional info, e.g. current score
        """

        # Apply the move
        r, c = action
        self.board[r, c] = self.current_player
        self.player_positions[self.current_player].append((r, c))

        if not self.first_move_done[self.current_player]:
            self.first_move_done[self.current_player] = True

        # Calculate reward for completing regions
        completed_regions = self._count_completed_regions_by_move((r, c))
        reward = self._region_completion_reward(completed_regions)

        # Update scores
        if self.current_player == 1:
            self.score = (self.score[0] + reward, self.score[1])
        else:
            self.score = (self.score[0], self.score[1] + reward)

        # After the move, check if the next player can move
        next_player = -self.current_player
        if self.get_all_moves(next_player):
            self.player_can_move[next_player] = True
        else:
            self.player_can_move[next_player] = False

        # Check if current player can still move
        if self.get_all_moves(self.current_player):
            self.player_can_move[self.current_player] = True
        else:
            self.player_can_move[self.current_player] = False

        # Switch player if the next player can move
        if self.player_can_move[next_player]:
            self.current_player = next_player
        else:
            # If next player cannot move, check if current player can still move
            if self.player_can_move[self.current_player]:
                # Current player gets another turn
                pass
            else:
                # Both players cannot move
                pass

        done = self.is_terminal()

        return reward, done, {"score": self.score}

    def _region_completion_reward(self, regions_completed):
        """Given the number of regions completed, return the corresponding reward."""
        if regions_completed == 0:
            return 0
        elif regions_completed == 1:
            return 1
        elif regions_completed == 2:
            return 3
        elif regions_completed == 3:
            return 7
        else:
            # Should not happen (there are at most 3 regions completed with one move: a row, a column, and a 3x3 block)
            return 0

    def _count_completed_regions_by_move(self, move):
        """Count how many regions (row, column, block) are completed by placing a piece at move=(r,c).
           A region is complete if there are no empty cells in that region."""
        (r, c) = move
        completed_count = 0

        # Check row r
        if self._is_region_complete_row(r):
            completed_count += 1

        # Check column c
        if self._is_region_complete_col(c):
            completed_count += 1

        # Check 3x3 block
        # Determine which 3x3 block (r,c) belongs to
        block_r = (r // 3) * 3
        block_c = (c // 3) * 3
        if self._is_region_complete_block(block_r, block_c):
            completed_count += 1

        return completed_count

    def _is_region_complete_row(self, row):
        """Check if given row is fully occupied (no zero)."""
        return np.all(self.board[row, :] != 0)

    def _is_region_complete_col(self, col):
        """Check if given col is fully occupied (no zero)."""
        return np.all(self.board[:, col] != 0)

    def _is_region_complete_block(self, start_r, start_c):
        """Check if the 3x3 block starting at (start_r, start_c) is fully occupied."""
        block = self.board[start_r:start_r+3, start_c:start_c+3]
        return np.all(block != 0)

    def clone(self):
        """Return a deep copy of the current state."""
        new_state = DQLGameState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.score = (self.score[0], self.score[1])
        new_state.first_move_done = {1: self.first_move_done[1], -1: self.first_move_done[-1]}
        new_state.player_positions = {
            1: list(self.player_positions[1]),
            -1: list(self.player_positions[-1])
        }
        new_state.player_can_move = {
            1: self.player_can_move[1],
            -1: self.player_can_move[-1]
        }
        return new_state
