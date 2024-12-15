import numpy as np
import random
import torch
import pickle
from env2x2 import DQLGameState
from DQN2x2 import SimpleCNNQNetwork

def load_model(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    model = SimpleCNNQNetwork(input_shape=(4,4), num_actions=16)
    model.load_state_dict(data["policy_state_dict"])
    model.eval()
    print(f"Model loaded from {filename}")
    return model

def make_state(board):
    """Create the one-hot encoded state representation."""
    p1_channel = (board == 1).astype(np.float32)
    p2_channel = (board == -1).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,4,4)

def select_max_q_action(model, state, valid_moves):
    """Select the action with the highest Q-value."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 3, 4, 4)
    with torch.no_grad():
        q_values = model(state_t).squeeze(0).numpy()  # (16,)
    
    # Mask invalid moves
    q_values_masked = np.full_like(q_values, -1e9)
    for r, c in valid_moves:
        action_idx = r * 4 + c
        q_values_masked[action_idx] = q_values[action_idx]
    
    best_action_idx = np.argmax(q_values_masked)
    return best_action_idx // 4, best_action_idx % 4

def random_opponent_move(state):
    """Generate a random valid move for the opponent."""
    moves = state.get_all_moves()
    if len(moves) == 0:
        return None
    return random.choice(moves)

def test_model(model, num_games=10):
    """Test the trained model against a random opponent."""
    env = DQLGameState()
    results = []
    scores = []

    for game in range(num_games):
        env.reset()
        done = False
        agent_score = 0
        opponent_score = 0

        while not done:
            valid_moves = env.get_all_moves()
            if len(valid_moves) == 0:
                break

            # Agent move
            current_board = env.board.copy()
            state = make_state(current_board)
            action = select_max_q_action(model, state, valid_moves)
            reward, done, info = env.step(action)
            agent_score = info["score"][0]
            opponent_score = info["score"][1]

            if done:
                break

            # Opponent move (random)
            opponent_action = random_opponent_move(env)
            if opponent_action is not None:
                _, done, info = env.step(opponent_action)
                agent_score = info["score"][0]
                opponent_score = info["score"][1]

        # Store the results
        result = "Win" if agent_score > opponent_score else "Loss" if agent_score < opponent_score else "Draw"
        results.append(result)
        scores.append((agent_score, opponent_score))
        print(f"Game {game + 1}: {result} (Agent: {agent_score}, Opponent: {opponent_score})")

    # Print summary
    print("\nSummary:")
    print(f"Games played: {num_games}")
    print(f"Wins: {results.count('Win')}")
    print(f"Losses: {results.count('Loss')}")
    print(f"Draws: {results.count('Draw')}")

if __name__ == "__main__":
    model = load_model("C:\\Users\\20203734\\OneDrive - TU Eindhoven\\2AMU10_Group35_Sudoku\\Sudoku environment\\2x2\\models\\dqn_model_2x2.pkl")
    test_model(model, num_games=1000)
