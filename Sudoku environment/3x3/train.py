# train.py
import random
import numpy as np
import time
import os
import torch
import pickle
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from env import DQLGameState  
from DQN import DQNAgent      
from test import plot_board
from tqdm import tqdm
from opponents import select_action_score_or_mobility, random_opponent_move, select_action_score

def check_device():
    """
    Checks if a GPU is available and prints device information.
    If no GPU is found, it falls back to CPU.
    """
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU is NOT available. Using CPU.")


def load_model(agent, filename="dqn_model.pkl"):
    """
    Loads a saved model (pickled) and initializes the agent with the loaded weights.
    Synchronizes the target network as well.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    file_path = os.path.join(models_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file '{file_path}' not found.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    agent.policy_net.load_state_dict(data["policy_state_dict"])
    agent.target_net.load_state_dict(data["policy_state_dict"])  # Synchronize target_net
    print(f"Model loaded from {file_path}.")


def save_model_as_pkl(agent, epsilon, filename="dqn_model.pkl"):
    """
    Saves the current model as a .pkl file. 
    Includes the policy network state_dict and the current epsilon.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "models")
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_time = filename.replace(".pkl", f"_{timestamp}.pkl")

    # If you want unique versions with timestamps, use 'filename_with_time'
    filepath = os.path.join(directory, filename)

    data_to_save = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }

    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model and parameters saved to {filepath}")


def make_state(board, player=1):
    """
    Creates a 3-channel representation from the perspective of 'player'.
    Board shape: (9, 9) with values in {-1, 0, +1}.
    
    Channel 0: cells owned by 'player'
    Channel 1: cells owned by '-player'
    Channel 2: empty cells
    """
    p1_channel = (board == player).astype(np.float32)
    p2_channel = (board == -player).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3, 9, 9)


def selfplay(state_obj, opponent_agent=None, player=-1):
    """
    Used for self-play scenarios. Fetches valid moves and forms the opponent's state
    to let the opponent_agent pick an action.
    """
    valid_moves = state_obj.get_all_moves()
    opp_board = state_obj.board.copy().astype(np.float32)
    opp_state = make_state(opp_board, player=player)
    return opponent_agent.select_action(opp_state, valid_moves, epsilon=0)


def main():
    """
    Main training loop for the DQNAgent on a 9x9 board environment.
    
    - Checks device (GPU/CPU).
    - Instantiates the agent.
    - Tries to load a pre-trained model.
    - Trains for a specified number of episodes against an opponent agent.
    - Implements an epsilon-decay schedule in two phases.
    - Tracks statistics like win rate and average reward.
    - Saves the best model and plots training curves.
    """
    check_device()
    num_episodes = 50_000

    # Initialize agent (adjust replay_size, etc., as desired)
    agent = DQNAgent(lr=1e-3, gamma=0.99, batch_size=64, replay_size=50_000, update_target_every=500)

    # Normalization factor for rewards
    normalization_factor = 7.0

    # Try loading an existing model
    model_filename = "team35_9x9_dqn_model.pkl"  # Adjust to your own model filename
    try:
        load_model(agent, filename=model_filename)
        print("Starting training with preloaded model.")
    except FileNotFoundError:
        print("No pretrained model found. Starting training from scratch.")

    # Opponent agent (greedy or mobility policy)
    opponent_agent = copy.deepcopy(agent)

    # Epsilon-decay in two phases
    epsilon_start = 1
    epsilon_mid = 0.5
    epsilon_end = 0
    epsilon_decay_steps_phase1 = 25_000
    epsilon_decay_steps_phase2 = 25_000
    total_decay_steps = epsilon_decay_steps_phase1 + epsilon_decay_steps_phase2

    epsilon_step_phase1 = (epsilon_start - epsilon_mid) / epsilon_decay_steps_phase1
    epsilon_step_phase2 = (epsilon_mid - epsilon_end) / epsilon_decay_steps_phase2

    # Statistics
    win_rate_history = []
    avg_rewards_per_interval = []
    win_rate_interval = 500

    win_count = 0
    episodes_tracked = 0
    total_rewards = 0.0
    rounds_in_interval = 0
    best_win_rate = 0.0

    # Environment
    state_obj = DQLGameState()

    # Start training
    for episode in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
        # --- Determine epsilon ---
        if episode < epsilon_decay_steps_phase1:
            epsilon = max(epsilon_mid, epsilon_start - episode * epsilon_step_phase1)
        elif episode < total_decay_steps:
            episodes_in_phase2 = episode - epsilon_decay_steps_phase1
            epsilon = max(epsilon_end, epsilon_mid - episodes_in_phase2 * epsilon_step_phase2)
        else:
            epsilon = epsilon_end

        # --- Randomly decide if the agent is player +1 or -1 ---
        if random.random() < 0.5:
            agent_player = 1
        else:
            agent_player = -1
        opponent_player = -agent_player

        opponent_policy = select_action_score_or_mobility

        # Reset board
        state_obj.reset()

        # --- If opponent_player starts, let them make the opening move ---
        if state_obj.current_player == opponent_player:
            opponent_action = opponent_policy(state_obj, opponent_agent=opponent_agent, player=opponent_player)
            valid_moves = state_obj.get_all_moves()
            reward_opponent, done_opponent, info = state_obj.step(opponent_action)

        # plot_board(state_obj.board, title= 'agent is: '+str(agent_player), pause_time=3)

        done = False
        episode_reward = 0.0

        final_agent_score = 0
        final_opponent_score = 0

        # Loop until the game is over
        while not done:
            if state_obj.current_player == agent_player:
                # --- AGENT TURN ---
                valid_moves = state_obj.get_all_moves(player=agent_player)
                agent_valid_moves_before = valid_moves

                current_board = state_obj.board.copy().astype(np.float32)
                current_state = make_state(current_board, player=agent_player)

                action = agent.select_action(current_state, valid_moves, epsilon)
                reward_agent, done_agent, info = state_obj.step(action)
                # plot_board(state_obj.board)
                reward_agent /= normalization_factor

                # Score in info["score"] = (score_p1, score_p2)
                if agent_player == 1:
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]
                else:
                    final_agent_score = info["score"][1]
                    final_opponent_score = info["score"][0]

                if done_agent:
                    # Store and update
                    next_board = state_obj.board.copy().astype(np.float32)
                    next_state = make_state(next_board, player=agent_player)
                    agent.store_transition(current_state, action, reward_agent, next_state, True)
                    agent.update()
                    break

                # Check if the opponent (opponent_player) has any moves left
                valid_moves_opponent = state_obj.get_all_moves(player=opponent_player)
                if not valid_moves_opponent:
                    # If the opponent cannot move, let the agent play until the game is finished (playout)
                    next_state = make_state(state_obj.board, player=agent_player)
                    cum_reward = 0.0
                    done_playout = False
                    nr_moves = 0 

                    while not done_playout:
                        # plot_board(state_obj.board)
                        # Agent continues to play until the game is over
                        playout_state = make_state(state_obj.board, player=opponent_player)
                        playout_moves = state_obj.get_all_moves(player=opponent_player)
                        if not playout_moves:
                            # No moves left for agent -> game is likely done
                            break
                        playout_action = agent.select_action(playout_state, playout_moves, epsilon)
                        rew, done_playout, info_playout = state_obj.step(playout_action)
                        nr_moves += 1
                        # plot_board(state_obj.board)
                        rew /= normalization_factor
                        # Optional bonus for mobility can be added here
                        cum_reward += rew
                        cum_reward += (nr_moves * 1) / normalization_factor  # Encourage not playing in already claimed territory

                        if agent_player == 1:
                            final_agent_score = info_playout["score"][0]
                            final_opponent_score = info_playout["score"][1]
                        else:
                            final_agent_score = info_playout["score"][1]
                            final_opponent_score = info_playout["score"][0]

                        if done_playout:
                            combined_reward = reward_agent + cum_reward
                            # plot_board(state_obj.board, title= str(combined_reward))
                            agent.store_transition(current_state, action, combined_reward, next_state, True)
                            agent.update()
                            break
                    break  # Proceed to the next episode
                # Else continue, environment switches to opponent_player
                continue

            else:
                # --- OPPONENT TURN ---
                opp_moves = state_obj.get_all_moves()
                opponent_valid_moves_before = opp_moves

                opponent_action = opponent_policy(state_obj, opponent_agent=opponent_agent, player=opponent_player)
                reward_opponent, done_opponent, info = state_obj.step(opponent_action)
                # plot_board(state_obj.board)
                reward_opponent /= normalization_factor

                # Update final scores
                if agent_player == 1:
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]
                else:
                    final_agent_score = info["score"][1]
                    final_opponent_score = info["score"][0]

                combined_reward = reward_agent - reward_opponent

                if done_opponent:
                    episode_reward += combined_reward
                    current_board = state_obj.board.copy().astype(np.float32)
                    current_state = make_state(current_board, player=agent_player)
                    agent.store_transition(current_state, action, combined_reward, current_state, True)
                    agent.update()
                    break

                # --- Agent moves after the opponent (mobility reward) ---
                valid_moves_agent = state_obj.get_all_moves(player=agent_player)
                if valid_moves_agent:
                    agent_valid_moves_after = valid_moves_agent
                    opponent_valid_moves_after = state_obj.get_all_moves(player=opponent_player)

                    # Calculate mobility reward: agent's gain minus opponent's gain
                    mobility_reward = 0.05 * (len(agent_valid_moves_after) - len(opponent_valid_moves_after))
                    mobility_reward /= normalization_factor

                    combined_reward += mobility_reward
                    episode_reward += combined_reward

                    # Store transition
                    current_board = state_obj.board.copy().astype(np.float32)
                    next_state = make_state(current_board, player=agent_player)
                    agent.store_transition(current_state, action, combined_reward, next_state, False)
                    # plot_board(state_obj.board, title=str((mobility_reward,combined_reward)), pause_time=1.5)
                    agent.update()

                else:
                    # Agent has no moves left, so the opponent plays until the game ends
                    cum_reward = 0.0
                    done_playout = False
                    while not done_playout:
                        opp_action = opponent_policy(state_obj, opponent_agent=opponent_agent, player=opponent_player)

                        rew_opp, done_playout, info_p = state_obj.step(opp_action)
                        rew_opp /= normalization_factor
                        cum_reward += rew_opp
                        if agent_player == 1:
                            final_agent_score = info_p["score"][0]
                            final_opponent_score = info_p["score"][1]
                        else:
                            final_agent_score = info_p["score"][1]
                            final_opponent_score = info_p["score"][0]

                        if done_playout:
                            # combined_reward = (previous) - cum_reward
                            combined_reward = combined_reward - cum_reward
                            current_board = state_obj.board.copy().astype(np.float32)
                            current_state = make_state(current_board, player=agent_player)
                            agent.store_transition(current_state, action, combined_reward, current_state, True)
                            agent.update()

                            break
                    break
                continue

        # End of episode: check if the agent won
        if final_agent_score > final_opponent_score:
            win_count += 1
        episodes_tracked += 1

        total_rewards += episode_reward
        rounds_in_interval += 1

        # Log stats every 'win_rate_interval' episodes
        if (episode + 1) % win_rate_interval == 0:
            win_rate = win_count / episodes_tracked
            win_rate_history.append(win_rate)
            avg_reward = total_rewards / rounds_in_interval
            avg_rewards_per_interval.append(avg_reward)

            # Save the model if it has the best win_rate
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_model_as_pkl(agent, best_win_rate, filename="9x9_self_best_dqn_model.pkl")

            print(f"Episode {episode+1}/{num_episodes} completed. Epsilon: {epsilon:.3f}")
            print(f"Win Rate (last {win_rate_interval} eps): {win_rate:.2f}")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Last Episode Score: (agent={final_agent_score}, opp={final_opponent_score})")

            # Reset counters
            win_count = 0
            episodes_tracked = 0
            total_rewards = 0.0
            rounds_in_interval = 0

    print("Training completed.")
    save_model_as_pkl(agent, epsilon)

    # Plotting
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot win rate
    plt.figure(figsize=(10, 6))
    plt.plot(range(win_rate_interval, num_episodes + 1, win_rate_interval), win_rate_history, label="Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Training")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(plots_dir, "win_rate_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Win rate plot saved to {plot_path}")

    # Plot average reward
    plt.figure(figsize=(10, 6))
    plt.plot(range(win_rate_interval, num_episodes + 1, win_rate_interval), avg_rewards_per_interval, label="Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Interval Over Training")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(plots_dir, "average_reward_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Average reward plot saved to {plot_path}")


if __name__ == "__main__":
    main()
