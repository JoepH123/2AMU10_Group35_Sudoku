# env_6x6.py

import numpy as np

class DQLGameState6x6:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game state to its initial state for a 6x6 board."""
        self.board = np.zeros((6, 6), dtype=int)
        self.current_player = 1
        self.score = (0, 0)
        self.first_move_done = {1: False, -1: False}
        self.player_positions = {1: [], -1: []}
        self.player_can_move = {1: True, -1: True}

    def get_all_moves(self, player=None):
        """Return a list of all possible moves (row, col) for the specified player."""
        if player is None:
            player = self.current_player

        # Als player_can_move[player] False is, return []
        if not self.player_can_move[player]:
            return []

        moves = []
        if not self.first_move_done[player]:
            # Eerste zet:
            # Player 1 in row=0, Player -1 in row=5
            start_row = 0 if player == 1 else 5
            for c in range(6):
                if self.board[start_row, c] == 0:
                    moves.append((start_row, c))
            return moves

        # Vervolgzetten: adjacent aan reeds geplaatste eigen stenen
        occupied_positions = self.player_positions[player]
        if not occupied_positions:
            return []

        possible_moves = set()
        for (r, c) in occupied_positions:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 6 and 0 <= nc < 6:
                        if self.board[nr, nc] == 0:
                            possible_moves.add((nr, nc))
        return list(possible_moves)

    def is_terminal(self):
        """Game is over if both players cannot move."""
        return not (self.player_can_move[1] or self.player_can_move[-1])

    def step(self, action):
        """Perform action (r,c) for current_player."""
        r, c = action
        self.board[r, c] = self.current_player
        self.player_positions[self.current_player].append((r, c))

        if not self.first_move_done[self.current_player]:
            self.first_move_done[self.current_player] = True

        # Count completed items (row, col, 2×3 block)
        completed_count = 0
        if self._is_row_complete(r):
            completed_count += 1
        if self._is_col_complete(c):
            completed_count += 1
        # Bepaal start van 2×3 blok
        block_r = (r // 2) * 2  # 2-rij-blok
        block_c = (c // 3) * 3  # 3-col-blok
        if self._is_region_complete_2x3(block_r, block_c):
            completed_count += 1

        # Reward
        reward = self._region_completion_reward(completed_count)

        # Update score
        if self.current_player == 1:
            self.score = (self.score[0] + reward, self.score[1])
        else:
            self.score = (self.score[0], self.score[1] + reward)

        # Check next player
        next_player = -self.current_player
        moves_next = self.get_all_moves(next_player)
        if moves_next:
            self.player_can_move[next_player] = True
        else:
            self.player_can_move[next_player] = False

        # Check current player
        moves_curr = self.get_all_moves(self.current_player)
        if moves_curr:
            self.player_can_move[self.current_player] = True
        else:
            self.player_can_move[self.current_player] = False

        # Wissel indien de next player kan
        if self.player_can_move[next_player]:
            self.current_player = next_player
        else:
            # Als next player niet kan, check of current player kan
            # Zo niet, game eindigt
            pass

        done = self.is_terminal()
        return reward, done, {"score": self.score}

    def _is_row_complete(self, row):
        """Check if row is fully occupied."""
        return np.all(self.board[row, :] != 0)

    def _is_col_complete(self, col):
        """Check if col is fully occupied."""
        return np.all(self.board[:, col] != 0)

    def _is_region_complete_2x3(self, start_r, start_c):
        """Check if the 2x3 block is fully occupied."""
        block = self.board[start_r:start_r+2, start_c:start_c+3]
        return np.all(block != 0)

    def _region_completion_reward(self, items_completed):
        """1,3,7 points for completing 1,2,3 items at once."""
        if items_completed == 0:
            return 0
        elif items_completed == 1:
            return 1
        elif items_completed == 2:
            return 3
        elif items_completed == 3:
            return 7
        return 0

    def clone(self):
        """Deep copy of the state."""
        new_state = DQLGameState6x6()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.score = (self.score[0], self.score[1])
        new_state.first_move_done = {
            1: self.first_move_done[1],
            -1: self.first_move_done[-1]
        }
        new_state.player_positions = {
            1: list(self.player_positions[1]),
            -1: list(self.player_positions[-1])
        }
        new_state.player_can_move = {
            1: self.player_can_move[1],
            -1: self.player_can_move[-1]
        }
        return new_state
