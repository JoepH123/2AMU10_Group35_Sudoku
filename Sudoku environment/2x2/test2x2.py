import numpy as np
import random
import torch
import pickle
import time
import os
import matplotlib.pyplot as plt
from env2x2 import DQLGameState
from DQN2x2 import SimpleCNNQNetwork

def load_model(filename="dqn_model_2x2.pkl"):
    """Load the trained model from the 'models' folder in the same directory."""
    # Dynamisch pad naar de 'models' folder bepalen
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Map van dit script
    models_dir = os.path.join(script_dir, "models")  # Pad naar 'models' map
    file_path = os.path.join(models_dir, filename)  # Volledige pad naar modelbestand

    # Model laden
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    model = SimpleCNNQNetwork(input_shape=(4, 4), num_actions=16)
    model.load_state_dict(data["policy_state_dict"])
    model.eval()
    print(f"Model loaded from {file_path}")
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

def plot_board(board, title="Game State", pause_time=1.5):
    """Update the existing plot with the current game board in one static figure."""
    plt.clf()  # Clear the current figure
    
    N = 4  # Board size
    ax = plt.gca()  # Get current Axes to keep it in one figure

    # Draw the board with colored cells
    ax.imshow(board, cmap="coolwarm", vmin=-1, vmax=1, origin="upper")

    # Draw thin gridlines for all cells
    for x in range(N + 1):
        ax.axhline(x - 0.5, color="black", linewidth=1)  # Horizontal lines
        ax.axvline(x - 0.5, color="black", linewidth=1)  # Vertical lines

    # Draw thick lines for 2x2 regions
    for x in range(0, N + 1, 2):
        ax.axhline(x - 0.5, color="black", linewidth=2)  # Horizontal 2x2 borders
        ax.axvline(x - 0.5, color="black", linewidth=2)  # Vertical 2x2 borders

    # Add text in each cell
    for i in range(N):
        for j in range(N):
            value = board[i, j]
            if value == 1:
                text = "P1"
                color = "white"
            elif value == -1:
                text = "P2"
                color = "white"
            else:
                text = "."
                color = "gray"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, color=color)

    # Set title and subtitle (live score or game info)
    plt.title(title, fontsize=14)
    plt.xlabel("Player 1 (P1) vs Player 2 (P2)", fontsize=10, labelpad=10)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Update the plot without creating a new figure
    plt.draw()
    plt.pause(pause_time)  # Pause to allow viewing



def test_model(model, num_games=10):
    """Test the trained model against a random opponent."""
    env = DQLGameState()
    wins = 0

    for game in range(num_games):
        env.reset()
        done = False
        agent_score = 0
        opponent_score = 0
        print(f"\nGame {game + 1}:")

        while not done:
            # Visualize current board state
            plot_board(env.board, title=f"Agent's Move (Score: {agent_score} - {opponent_score})")

            # Agent move
            valid_moves = env.get_all_moves()
            if len(valid_moves) == 0:
                # Check opponent can move; otherwise, game over
                opponent_moves = env.get_all_moves(player=-env.current_player)
                if len(opponent_moves) == 0:
                    done = True
                    break
                env.current_player = -env.current_player  # Switch to opponent
                continue

            current_board = env.board.copy()
            state = make_state(current_board)
            action = select_max_q_action(model, state, valid_moves)
            reward, done, info = env.step(action)
            agent_score = info["score"][0]
            opponent_score = info["score"][1]

            if done:
                break

            # Visualize after opponent move
            plot_board(env.board, title=f"Opponent's Move (Score: {agent_score} - {opponent_score})")

            # Opponent move (random)
            opponent_moves = env.get_all_moves()
            if len(opponent_moves) == 0:
                # Check agent can move; otherwise, game over
                agent_moves = env.get_all_moves(player=-env.current_player)
                if len(agent_moves) == 0:
                    done = True
                    break
                env.current_player = -env.current_player  # Switch back to agent
                continue

            opponent_action = random.choice(opponent_moves)
            _, done, info = env.step(opponent_action)
            agent_score = info["score"][0]
            opponent_score = info["score"][1]

        # Final board state
        plot_board(env.board, title=f"Final State (Agent: {agent_score}, Opponent: {opponent_score})")
        if agent_score > opponent_score:
            wins += 1

        result = "Win" if agent_score > opponent_score else "Loss" if agent_score < opponent_score else "Draw"
        print(f"Game {game + 1}: {result} (Agent: {agent_score}, Opponent: {opponent_score})")

    return wins/num_games



if __name__ == "__main__":
    start_time = time.time()  # Start de timer
    model = load_model(filename='best_dqn_model.pkl')
    end_time = time.time()  # Stop de timer
    load_duration = end_time - start_time
    print(f"Model loading time: {load_duration:.4f} seconds")

    print(test_model(model, num_games=10))