import random

def random_opponent_move(state):
    moves = state.get_all_moves()
    if len(moves) == 0:
        return None
    return random.choice(moves)


def calculate_mobility(state, player=-1):
    state.current_player = player # after move look as if it was our move to see mobility
    return len(state.get_all_moves())


def select_action_score_or_mobility(state, player=-1):
    """
    Selecteert een actie die punten scoort; als er geen acties punten scoren,
    selecteert de actie met maximale mobiliteit.

    Parameters:
    - state: de huidige staat van het spel
    - valid_moves: lijst van geldige zetten
    - player: de speler die de zet uitvoert

    Returns:
    - De beste zet op basis van scoren of mobiliteit
    """
    valid_moves = state.get_all_moves()
    max_mobility = -1
    best_moves_mobility = []
    best_scoring_moves = []

    for move in valid_moves:
        # Maak een kopie van de staat om te simuleren
        state_copy = state.clone()
        reward, done, _ = state_copy.step(move)

        # Controleer of de zet punten scoort
        if reward > 0:
            best_scoring_moves.append(move)

        # Bereken mobiliteit voor de zet
        mobility = calculate_mobility(state_copy, player)
        if mobility > max_mobility:
            max_mobility = mobility
            best_moves_mobility = [move]
        elif mobility == max_mobility:
            best_moves_mobility.append(move)

    # Kies een scorende zet als die bestaat, anders een zet met maximale mobiliteit
    if best_scoring_moves:
        return random.choice(best_scoring_moves)
    return random.choice(best_moves_mobility) if best_moves_mobility else random.choice(valid_moves)


def select_action_score(state):
    """
    Selecteert een actie die punten scoort; als er geen acties punten scoren,
    selecteert een random actie

    Parameters:
    - state: de huidige staat van het spel
    - valid_moves: lijst van geldige zetten
    - player: de speler die de zet uitvoert

    Returns:
    - De beste zet op basis van scoren 
    """
    valid_moves = state.get_all_moves()
    max_reward = 0
    best_scoring_moves = []

    for move in valid_moves:
        # Maak een kopie van de staat om te simuleren
        state_copy = state.clone()
        reward, done, _ = state_copy.step(move)

        if reward > max_reward:
            max_reward = reward
            best_scoring_moves = [move]
        elif reward == max_reward:
            best_scoring_moves.append(move)

    return random.choice(best_scoring_moves) if best_scoring_moves else random.choice(valid_moves)


