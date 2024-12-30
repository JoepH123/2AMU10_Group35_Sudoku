# train.py
import random
import numpy as np
import time
import os
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from env import DQLGameState  
from DQN import DQNAgent      
from test import plot_board
from tqdm import tqdm
from opponents import select_action_score_or_mobility, random_opponent_move, select_action_score


def check_device():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU is NOT available. Using CPU.")


def load_model(agent, filename="dqn_model.pkl"):
    """Laad een opgeslagen model en initialiseer de agent met de gewichten."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    file_path = os.path.join(models_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file '{file_path}' not found.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    agent.policy_net.load_state_dict(data["policy_state_dict"])
    agent.target_net.load_state_dict(data["policy_state_dict"])  # Synchroniseer target_net
    print(f"Model loaded from {file_path}.")


def save_model_as_pkl(agent, epsilon, filename="dqn_model.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "models")

    # Zorg ervoor dat de 'models' map bestaat
    os.makedirs(directory, exist_ok=True)

    # Voeg een tijdstempel toe aan de bestandsnaam (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_time = filename.replace(".pkl", f"_{timestamp}.pkl")

    filepath = os.path.join(directory, filename) # filename_with_time als je unieke wilt

    data_to_save = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }

    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model and parameters saved to {filepath}")


def make_state(board, player=1):
    # board shape: (4,4) with values in {-1,0,1}
    # We maken 3 kanalen:
    # kanaal 0: speler 1 posities
    # kanaal 1: speler -1 posities
    # kanaal 2: lege cellen
    p1_channel = (board == player).astype(np.float32)
    p2_channel = (board == -player).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,4,4)

def main():
    check_device()
    num_episodes = 50_000
    agent = DQNAgent(lr=5e-4, gamma=0.99, batch_size=128, replay_size=1000, update_target_every=1000)
    normalization_factor = 7.0

    # Laad het bestaande model (optioneel)
    model_filename = "9x9_greedy_1_best_dqn_model.pkl"  # Pas dit aan naar jouw opgeslagen modelbestand
    try:
        load_model(agent, filename=model_filename)  # Initialiseer met opgeslagen gewichten
        print("Starting training with preloaded model.")
    except FileNotFoundError:
        print("No pretrained model found. Starting training from scratch.")

    opponent_policy = select_action_score

    # Stel de parameters in
    epsilon_start = 0.3
    epsilon_mid = 0.1
    epsilon_end = 0.01

    epsilon_decay_steps_phase1 = 20_000
    epsilon_decay_steps_phase2 = 30_000
    total_decay_steps = epsilon_decay_steps_phase1 + epsilon_decay_steps_phase2

    epsilon_step_phase1 = (epsilon_start - epsilon_mid) / epsilon_decay_steps_phase1
    epsilon_step_phase2 = (epsilon_mid - epsilon_end) / epsilon_decay_steps_phase2

    win_rate_history = []
    avg_rewards_per_interval = []
    win_count = 0
    episodes_tracked = 0
    win_rate_interval = 500

    total_rewards = 0
    rounds_in_interval = 0

    state_obj = DQLGameState()

    best_win_rate = 0.0

    for episode in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):

        if episode < epsilon_decay_steps_phase1:  # Fase 1
            epsilon = max(epsilon_mid, epsilon_start - episode * epsilon_step_phase1)
        elif episode < total_decay_steps:         # Fase 2
            episodes_in_phase2 = episode - epsilon_decay_steps_phase1
            epsilon = max(epsilon_end, epsilon_mid - episodes_in_phase2 * epsilon_step_phase2)
        else:
            epsilon = epsilon_end

        state_obj.reset()
        #plot_board(state_obj.board)
        done = False
        final_agent_score = 0
        final_opponent_score = 0
        episode_reward = 0.0

        while not done:

            if state_obj.current_player == 1: # if agent to move
                valid_moves = state_obj.get_all_moves()
                agent_valid_moves_before = len(valid_moves)

                current_board = state_obj.board.copy().astype(np.float32)
                current_state = make_state(current_board)
                action = agent.select_action(current_state, valid_moves, epsilon)

                #plot_board(state_obj.board)
                reward_agent, done_agent, info = state_obj.step(action)
                reward_agent = reward_agent/normalization_factor
                #plot_board(state_obj.board)

                final_agent_score = info["score"][0]
                final_opponent_score = info["score"][1]

                if done_agent:
                    next_board = state_obj.board.copy().astype(np.float32)
                    next_state = make_state(next_board)
                    agent.store_transition(current_state, action, reward_agent, next_state, True)
                    #plot_board(state_obj.board, title=str(reward_agent))
                    agent.update()
                    break
                
                valid_moves_opponent = state_obj.get_all_moves(player=-1)
                opponent_valid_moves_before = len(valid_moves_opponent)

                if not valid_moves_opponent: # als opponent niet meer kan zetten, playout en cum reward toekenen van de laatste actie
                    cum_reward = 0
                    done_playout = False
                    next_state = make_state(state_obj.board)
                    while not done_playout:
                        #plot_board(state_obj.board)
                        new_state = make_state(state_obj.board)
                        valid_moves = state_obj.get_all_moves()
                        new_action = agent.select_action(new_state, valid_moves, epsilon)
                        reward, done_playout, info = state_obj.step(new_action)
                        cum_reward = cum_reward + (reward/normalization_factor) + ((len(valid_moves)*0.2)/normalization_factor)

                        final_agent_score = info["score"][0]
                        final_opponent_score = info["score"][1]
                        #plot_board(state_obj.board)

                        if done_playout:
                            combined_reward = reward_agent + cum_reward
                            #plot_board(state_obj.board, title=str(combined_reward))
                            agent.store_transition(current_state, action, combined_reward, next_state, True)
                            agent.update()
                            break
                    break
                continue


            else: # Opponent turn
                opponent_action = opponent_policy(state_obj)
                reward_opponent, done_opponent, info = state_obj.step(opponent_action)
                reward_opponent = reward_opponent/normalization_factor
                #plot_board(state_obj.board)

                final_agent_score = info["score"][0]
                final_opponent_score = info["score"][1]

                next_board = state_obj.board.copy().astype(np.float32)
                next_state = make_state(next_board)

                combined_reward = reward_agent - reward_opponent

                if done_opponent: # als het na move opponent klaar is
                    episode_reward += combined_reward
                    agent.store_transition(current_state, action, combined_reward, next_state, True)
                    #plot_board(state_obj.board, title=str(combined_reward))
                    agent.update()
                    break

                valid_moves_agent = state_obj.get_all_moves(player=1)
                if valid_moves_agent: # indien gewoon normaal doorgespeeld kan worden, nu dit opslaan
                    agent_valid_moves_after = len(valid_moves_agent)
                    opponent_valid_moves_after = len(state_obj.get_all_moves(player=-1))
                    mobility_reward = (((agent_valid_moves_after-agent_valid_moves_before) - (opponent_valid_moves_after-opponent_valid_moves_before))*0.2)/normalization_factor
                    combined_reward += mobility_reward
                    episode_reward += combined_reward
                    agent.store_transition(current_state, action, combined_reward, next_state, False)
                    #plot_board(state_obj.board, title='mobility reward: '+str(mobility_reward))
                    agent.update()

                
                elif not valid_moves_agent: # als Agent niet meer kan zetten doorspelen tot einde en cum_reward gebruiken van opponent 

                    cum_reward = 0
                    done_playout = False
                    while not done_playout:
                        #plot_board(state_obj.board)
                        action = opponent_policy(state_obj)
                        reward_opponent, done_playout, info = state_obj.step(action)
                        cum_reward += (reward_opponent/normalization_factor)

                        final_agent_score = info["score"][0]
                        final_opponent_score = info["score"][1]
                        #plot_board(state_obj.board)

                        if done_playout:
                            combined_reward = combined_reward - cum_reward
                            #plot_board(state_obj.board, title=str(combined_reward))
                            agent.store_transition(current_state, action, combined_reward, next_state, True)
                            agent.update()
                            break
                    break                    
                continue


        if final_agent_score > final_opponent_score:
            win_count += 1
        episodes_tracked += 1

        total_rewards += episode_reward
        rounds_in_interval += 1

        if (episode + 1) % win_rate_interval == 0:
            win_rate = win_count / episodes_tracked
            win_rate_history.append(win_rate)
            avg_reward = total_rewards / rounds_in_interval
            avg_rewards_per_interval.append(avg_reward)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_model_as_pkl(agent, best_win_rate, filename="9x9_greedy_best_dqn_model.pkl")

            if (episode + 1) % (win_rate_interval) == 0:
                print(f"Episode {episode+1}/{num_episodes} completed. Epsilon: {epsilon:.3f}")
                print(f"Win Rate over last {win_rate_interval} episodes: {win_rate:.2f}%")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Last Episode Score: {state_obj.score}")

            win_count = 0
            episodes_tracked = 0
            total_rewards = 0.0
            rounds_in_interval = 0

    print("Training completed.")
    save_model_as_pkl(agent, epsilon)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

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

