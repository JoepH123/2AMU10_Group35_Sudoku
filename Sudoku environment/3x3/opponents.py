import random

def random_opponent_move(state):
    """
    Selects a random move for the opponent from all valid moves.

    Parameters:
    -----------
    state : object
        The current game state. Must implement `get_all_moves()` returning valid moves.

    Returns:
    --------
    tuple or None
        A random valid move (e.g. (row, col)), or None if no moves are available.
    """
    moves = state.get_all_moves()
    if len(moves) == 0:
        return None
    return random.choice(moves)


def calculate_mobility(state, player=-1):
    """
    Calculates the mobility for a given player.
    Temporarily sets the `current_player` to simulate the move's effect on mobility.

    Parameters:
    -----------
    state : object
        The current game state. Must implement `get_all_moves()`, `step()`, etc.
    player : int, optional
        The player for which mobility is calculated, defaults to -1.

    Returns:
    --------
    int
        The number of valid moves for the specified player.
    """
    state.current_player = player
    return len(state.get_all_moves())


def select_action_score_or_mobility(state, opponent_agent=None, prob_random=0.1, player=-1):
    """
    Selects an action that either yields a positive reward ("scores") or, 
    if none do, picks the action(s) that maximize mobility. 
    With probability `prob_random`, a completely random move is chosen.

    Parameters:
    -----------
    state : object
        The current game state.
    opponent_agent : object, optional
        (Unused here) an opponent agent if needed for more complex calculations.
    prob_random : float, optional
        Probability of choosing a completely random action (for exploration).
    player : int, optional
        The player for which we are selecting a move (defaults to -1).

    Returns:
    --------
    tuple
        A valid move (row, col) that either produces a score or maximizes mobility.
    """
    valid_moves = state.get_all_moves()
    max_mobility = -1
    best_moves_mobility = []
    best_scoring_moves = []

    for move in valid_moves:
        # Clone and simulate the move
        state_copy = state.clone()
        reward, done, _ = state_copy.step(move)

        # Check if the move scores positively
        if reward > 0:
            best_scoring_moves.append(move)

        # Calculate mobility for this move
        mobility = calculate_mobility(state_copy, player)
        if mobility > max_mobility:
            max_mobility = mobility
            best_moves_mobility = [move]
        elif mobility == max_mobility:
            best_moves_mobility.append(move)

    # Prioritize a positively scoring move if available
    if best_scoring_moves:
        return random.choice(best_scoring_moves)
    else:
        # Otherwise, possibly pick a random move with probability `prob_random`
        if random.random() < prob_random:
            return random.choice(valid_moves)
        return random.choice(best_moves_mobility) if best_moves_mobility else random.choice(valid_moves)


def select_action_score(state, opponent_agent=None, player=None):
    """
    Greedy policy. Selects an action that yields the highest immediate reward; if no move 
    provides a positive reward, picks a random valid move.

    Parameters:
    -----------
    state : object
        The current game state.
    opponent_agent : object, optional
        (Unused here) an opponent agent if needed for more complex calculations.
    player : int, optional
        The player for which we are selecting a move (not used in this function).

    Returns:
    --------
    tuple
        A valid move (row, col) that produces the highest possible reward.
    """
    valid_moves = state.get_all_moves()
    max_reward = 0
    best_scoring_moves = []

    for move in valid_moves:
        # Clone and simulate the move
        state_copy = state.clone()
        reward, done, _ = state_copy.step(move)

        if reward > max_reward:
            max_reward = reward
            best_scoring_moves = [move]
        elif reward == max_reward:
            best_scoring_moves.append(move)

    # Return a best-scoring move if found, otherwise random
    return random.choice(best_scoring_moves) if best_scoring_moves else random.choice(valid_moves)

