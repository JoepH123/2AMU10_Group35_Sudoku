import numpy as np

class DQLGameState:
    def __init__(self):
        self.reset()  # Reset for new game

    def reset(self):
        """Reset the game state to its initial state."""
        self.board = np.zeros((4, 4), dtype=int)  # Reset to 4x4 empty board
        self.current_player = 1  # Player 1 starts
        self.score = (0, 0)  # Reset scores for both players
        self.first_move_done = {1: False, -1: False}
        self.player_positions = {1: [], -1: []}
        self.player_can_move = {1: True, -1: True}

    def get_all_moves(self, player=None):
        if player is None:
            player = self.current_player

        if not self.player_can_move[player]:
            return []

        moves = []
        if not self.first_move_done[player]:
            # First move constraints:
            # Player 1 must choose from top row (row = 0)
            # Player -1 must choose from bottom row (row = 3)
            start_row = 0 if player == 1 else 3
            for c in range(4):
                if self.board[start_row, c] == 0:
                    moves.append((start_row, c))
            return moves

        # If first move is done, must choose a cell adjacent
        occupied_positions = self.player_positions[player]
        if not occupied_positions:
            return []

        possible_moves = set()
        for (r, c) in occupied_positions:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        if self.board[nr, nc] == 0:
                            possible_moves.add((nr, nc))

        return list(possible_moves)

    def is_terminal(self):
        return not (self.player_can_move[1] or self.player_can_move[-1])

    def step(self, action):
        r, c = action
        self.board[r, c] = self.current_player
        self.player_positions[self.current_player].append((r, c))

        if not self.first_move_done[self.current_player]:
            self.first_move_done[self.current_player] = True

        # Count completed regions
        completed_regions = self._count_completed_regions_by_move((r, c))
        reward = self._region_completion_reward(completed_regions)

        # Update scores
        if self.current_player == 1:
            self.score = (self.score[0] + reward, self.score[1])
        else:
            self.score = (self.score[0], self.score[1] + reward)

        # Switch to next player if possible
        next_player = -self.current_player
        if self.get_all_moves(next_player):
            self.player_can_move[next_player] = True
        else:
            self.player_can_move[next_player] = False

        if self.get_all_moves(self.current_player):
            self.player_can_move[self.current_player] = True
        else:
            self.player_can_move[self.current_player] = False

        if self.player_can_move[next_player]:
            self.current_player = next_player
        else:
            # If next player cannot move, check if current player can still move
            if self.player_can_move[self.current_player]:
                pass
            else:
                pass

        done = self.is_terminal()
        # Normaliseer reward
        #reward = reward / 100
        return reward, done, {"score": self.score}

    def _region_completion_reward(self, regions_completed):
        if regions_completed == 0:
            return 0
        elif regions_completed == 1:
            return 1
        elif regions_completed == 2:
            return 3
        elif regions_completed == 3:
            return 7
        else:
            return 0

    def _count_completed_regions_by_move(self, move):
        (r, c) = move
        completed_count = 0

        # Check row r complete?
        if self._is_region_complete_row(r):
            completed_count += 1

        # Check column c complete?
        if self._is_region_complete_col(c):
            completed_count += 1

        # Check 2x2 block
        # Determine which 2x2 block (r,c) is in
        block_r = (r // 2) * 2
        block_c = (c // 2) * 2
        if self._is_region_complete_block(block_r, block_c):
            completed_count += 1

        return completed_count

    def _is_region_complete_row(self, row):
        return np.all(self.board[row, :] != 0)

    def _is_region_complete_col(self, col):
        return np.all(self.board[:, col] != 0)

    def _is_region_complete_block(self, start_r, start_c):
        block = self.board[start_r:start_r+2, start_c:start_c+2]
        return np.all(block != 0)

    def clone(self):
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
