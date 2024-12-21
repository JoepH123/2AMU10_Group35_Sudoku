# train.py
import random
import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from env2x2 import DQLGameState  # Import je 4x4 environment
from DQN2x2 import DQNAgent      # Import je DQN agent
from test2x2 import plot_board

def random_opponent_move(state):
    moves = state.get_all_moves()
    if len(moves) == 0:
        return None
    return random.choice(moves)


def check_device():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU is NOT available. Using CPU.")


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


def make_state(board):
    # board shape: (4,4) with values in {-1,0,1}
    # We maken 3 kanalen:
    # kanaal 0: speler 1 posities
    # kanaal 1: speler -1 posities
    # kanaal 2: lege cellen
    p1_channel = (board == 1).astype(np.float32)
    p2_channel = (board == -1).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,4,4)

def main():
    check_device()
    num_episodes = 100_000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 60_000 # decay steps
    agent = DQNAgent(lr=1e-3, gamma=0.99, batch_size=64, replay_size=1000, update_target_every=500)

    epsilon = epsilon_start
    epsilon_step = (epsilon_start - epsilon_end)/epsilon_decay

    win_rate_history = []
    avg_rewards_per_interval = []
    win_count = 0
    episodes_tracked = 0
    win_rate_interval = 100

    total_rewards = 0
    rounds_in_interval = 0

    state_obj = DQLGameState()

    best_win_rate = 0.0

    for episode in range(num_episodes):
        state_obj.reset()
        #plot_board(state_obj.board)
        done = False
        steps = 0
        final_agent_score = 0
        final_opponent_score = 0
        episode_reward = 0.0

        while not done:
            extra_opponent_reward = 0.0

            valid_moves = state_obj.get_all_moves()
            # if len(valid_moves) == 0:
            #     # Agent cannot move
            #     other_player = -state_obj.current_player
            #     print('player switch')
            #     other_moves = state_obj.get_all_moves(player=other_player)
            #     if len(other_moves) == 0:
            #         # Both can't move
            #         done = True
            #         break
            #     else:
            #         print('agent cant make any moves, opponent can')
            #         state_obj.current_player = other_player
            #         extra_opponent_reward = 5
            #         episode_reward += extra_opponent_reward
            #         continue

            current_board = state_obj.board.copy().astype(np.float32)
            current_state = make_state(current_board)
            action = agent.select_action(current_state, valid_moves, epsilon)
            reward_agent, done_agent, info = state_obj.step(action)
            #plot_board(state_obj.board)

            final_agent_score = info["score"][0]
            final_opponent_score = info["score"][1]

            next_board = state_obj.board.copy().astype(np.float32)
            next_state = make_state(next_board)

            # if done_agent:
            #     # Geen tegenstanderzet
            #     done = True
            #     combined_reward = reward_agent
            #     # Win/lose check
            #     # if final_agent_score > final_opponent_score:
            #     #     combined_reward += 100.0
            #     # elif final_agent_score < final_opponent_score:
            #     #     combined_reward -= 100.0
            #     episode_reward += combined_reward
            #     combined_reward = combined_reward/12
            #     agent.store_transition(current_state, action, combined_reward, next_state, True)
            #     print('end of game')
            #     agent.update()
            #     break

            # Opponent turn
            opponent_action = random_opponent_move(state_obj)
            # if opponent_action is None:
            #     # Opponent can't move
            #     other_player = -state_obj.current_player
            #     print('player switch')
            #     other_moves = state_obj.get_all_moves(player=other_player)
            #     if len(other_moves) == 0:
            #         # Both can't move
            #         done = True
            #         combined_reward = reward_agent
            #         # if final_agent_score > final_opponent_score:
            #         #     combined_reward += 100.0
            #         # elif final_agent_score < final_opponent_score:
            #         #     combined_reward -= 100.0
            #         episode_reward += combined_reward
            #         combined_reward = combined_reward/12
            #         agent.store_transition(current_state, action, combined_reward, next_state, True)
            #         print('both cant move')
            #         agent.update()
            #         break
            #     else:
            #         extra_reward = 5
            #         combined_reward = reward_agent + extra_reward
            #         episode_reward += combined_reward
            #         combined_reward = combined_reward/12
            #         agent.store_transition(current_state, action, combined_reward, next_state, False)
            #         agent.update()
            #         print('opponent trapped')
            #         continue

            reward_opponent, done_opponent, info = state_obj.step(opponent_action)
            #plot_board(state_obj.board)
            done = done_opponent
            final_agent_score = info["score"][0]
            final_opponent_score = info["score"][1]

            combined_reward = reward_agent - reward_opponent - extra_opponent_reward
            if done:
                # if final_agent_score > final_opponent_score:
                #     combined_reward += 100.0
                # elif final_agent_score < final_opponent_score:
                #     combined_reward -= 100.0
                episode_reward += combined_reward
                combined_reward = combined_reward/12
                agent.store_transition(current_state, action, combined_reward, next_state, True)
                agent.update()
            else:
                episode_reward += combined_reward
                combined_reward = combined_reward/12
                agent.store_transition(current_state, action, combined_reward, next_state, False)
                agent.update()

            steps += 1

        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon -= epsilon_step

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
                save_model_as_pkl(agent, best_win_rate, filename="new_best_dqn_model.pkl")

            if (episode + 1) % (win_rate_interval*10) == 0:
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

