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
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")  # index van de GPU
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
    """Sla het huidige model op als .pkl bestand."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "models")
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_time = filename.replace(".pkl", f"_{timestamp}.pkl")

    filepath = os.path.join(directory, filename)  # Of filename_with_time als je unieke versies wilt

    data_to_save = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }

    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model and parameters saved to {filepath}")

def make_state(board, player=1):
    """
    board shape: (9,9) with values in { -1, 0, +1 }
    We maken 3 kanalen vanuit het perspectief van 'player':
      kanaal 0: cellen van 'player'
      kanaal 1: cellen van '-player'
      kanaal 2: lege cellen
    """
    p1_channel = (board == player).astype(np.float32)
    p2_channel = (board == -player).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,9,9)

def selfplay(state_obj, opponent_agent=None, player=-1):
    valid_moves = state_obj.get_all_moves()
    opp_board = state_obj.board.copy().astype(np.float32)
    opp_state = make_state(opp_board, player=player)
    return opponent_agent.select_action(opp_state, valid_moves, epsilon=0)


def main():
    check_device()
    num_episodes = 50_000

    # Initialiseer agent (pas replay_size, etc. aan naar wens)
    agent = DQNAgent(lr=5e-4, gamma=0.99, batch_size=64, replay_size=100000, update_target_every=250)

    # Normalisatiefactor voor beloningen, zoals in jouw code
    normalization_factor = 7.0

    # Probeer een bestaand model te laden
    model_filename = "9x9_greedy_3_best_dqn_model.pkl"  # Pas aan naar jouw modelbestand
    try:
        load_model(agent, filename=model_filename)
        print("Starting training with preloaded model.")
    except FileNotFoundError:
        print("No pretrained model found. Starting training from scratch.")

    # Tegenstander (greedy of mobility policy)
    opponent_agent = copy.deepcopy(agent)

    # Epsilon-decay: 2 fasen
    epsilon_start = 0.5
    epsilon_mid = 0.1
    epsilon_end = 0.01
    epsilon_decay_steps_phase1 = 25_000
    epsilon_decay_steps_phase2 = 25_000
    total_decay_steps = epsilon_decay_steps_phase1 + epsilon_decay_steps_phase2

    epsilon_step_phase1 = (epsilon_start - epsilon_mid) / epsilon_decay_steps_phase1
    epsilon_step_phase2 = (epsilon_mid - epsilon_end) / epsilon_decay_steps_phase2

    # Statistieken
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
        # --- Bepaal epsilon ---
        if episode < epsilon_decay_steps_phase1:
            epsilon = max(epsilon_mid, epsilon_start - episode * epsilon_step_phase1)
        elif episode < total_decay_steps:
            episodes_in_phase2 = episode - epsilon_decay_steps_phase1
            epsilon = max(epsilon_end, epsilon_mid - episodes_in_phase2 * epsilon_step_phase2)
        else:
            epsilon = epsilon_end

        # --- Bepaal of agent speler +1 of -1 is (random) ---
        if random.random() < 0.5:
            agent_player = 1
        else:
            agent_player = -1
        opponent_player = -agent_player

        opponent_policy = select_action_score_or_mobility

        # if random.random() < 0.8:
        #     opponent_policy = select_action_score_or_mobility
        # else:
        #     opponent_policy = selfplay

        # Reset board
        state_obj.reset()
        # --- Laat de opponent eerst spelen als hij begint ---
        if state_obj.current_player == opponent_player:
            # Opponent doet een openingszet

            opponent_action = opponent_policy(state_obj, opponent_agent=opponent_agent, player=opponent_player)
            valid_moves = state_obj.get_all_moves()
            reward_opponent, done_opponent, info = state_obj.step(opponent_action)

        #plot_board(state_obj.board, title= 'agent is: '+str(agent_player) + f'opponent {opponent_policy}', pause_time=5)
 
        done = False
        episode_reward = 0.0

        final_agent_score = 0
        final_opponent_score = 0

        # Loop tot game klaar is
        while not done:
            if state_obj.current_player == agent_player:
                # --- AGENT TURN ---
                valid_moves = state_obj.get_all_moves(player=agent_player)
                agent_valid_moves_before = valid_moves #len(valid_moves)

                current_board = state_obj.board.copy().astype(np.float32)
                current_state = make_state(current_board, player=agent_player)

                action = agent.select_action(current_state, valid_moves, epsilon)
                reward_agent, done_agent, info = state_obj.step(action)
                #plot_board(state_obj.board)
                # Beloning komt al vanuit 'current_player' perspectief (volgens env),
                # dus als agent_player = -1 is, krijg je daar direct de juiste beloning.
                # Normaliseren
                reward_agent /= normalization_factor

                # Score in info["score"] = (score_p1, score_p2)
                # Bepaal final_agent_score op basis van agent_player
                if agent_player == 1:
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]
                else:
                    final_agent_score = info["score"][1]
                    final_opponent_score = info["score"][0]

                if done_agent:
                    # Opslaan en update
                    next_board = state_obj.board.copy().astype(np.float32)
                    next_state = make_state(next_board, player=agent_player)
                    agent.store_transition(current_state, action, reward_agent, next_state, True)
                    agent.update()
                    break

                # Check of tegenstander (opponent_player) nog moves heeft
                valid_moves_opponent = state_obj.get_all_moves(player=opponent_player)
                if not valid_moves_opponent:
                    # Als opponent geen zetten heeft, doe 'playout' voor agent
                    next_state = make_state(state_obj.board, player=agent_player)
                    cum_reward = 0.0
                    done_playout = False
                    nr_moves = 0 

                    while not done_playout:
                        #plot_board(state_obj.board)
                        # Agent speelt gewoon verder tot spel klaar
                        playout_state = make_state(state_obj.board, player=opponent_player)
                        playout_moves = state_obj.get_all_moves(player=opponent_player)
                        if not playout_moves:
                            # Geen zetten meer voor agent -> spel is in principe klaar
                            # We kunnen door de step() nog wel done krijgen
                            break
                        playout_action = agent.select_action(playout_state, playout_moves, epsilon)
                        rew, done_playout, info_playout = state_obj.step(playout_action)
                        nr_moves += 1
                        #plot_board(state_obj.board)
                        rew /= normalization_factor
                        # Extra bonus voor mobility (optioneel, zie code hieronder)
                        # ...
                        cum_reward += rew
                        cum_reward += (nr_moves*1)/normalization_factor # learn to not play in already claimed territory 

                        if agent_player == 1:
                            final_agent_score = info_playout["score"][0]
                            final_opponent_score = info_playout["score"][1]
                        else:
                            final_agent_score = info_playout["score"][1]
                            final_opponent_score = info_playout["score"][0]

                        if done_playout:
                            combined_reward = reward_agent + cum_reward
                            #plot_board(state_obj.board, title= str(combined_reward))
                            agent.store_transition(current_state, action, combined_reward, next_state, True)
                            agent.update()
                            break
                    break  # Ga naar volgende episode
                # Ga verder, environment wisselt zelf naar opponent_player
                continue

            else:
                # --- OPPONENT TURN ---
                opp_moves = state_obj.get_all_moves()
                opponent_valid_moves_before = opp_moves #len(opp_moves)


                opponent_action = opponent_policy(state_obj, opponent_agent=opponent_agent, player=opponent_player)
                reward_opponent, done_opponent, info = state_obj.step(opponent_action)
                #plot_board(state_obj.board)
                reward_opponent /= normalization_factor

                # Update final scores
                if agent_player == 1:
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]
                else:
                    final_agent_score = info["score"][1]
                    final_opponent_score = info["score"][0]

                combined_reward = reward_agent - 1.5*reward_opponent

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
                    #print(valid_moves_agent)
                    agent_valid_moves_after = valid_moves_agent #len(valid_moves_agent)
                    opponent_valid_moves_after = state_obj.get_all_moves(player=opponent_player) #len(state_obj.get_all_moves(player=opponent_player))

                    agent_control_before = len(set(agent_valid_moves_before) - set(opponent_valid_moves_before)) # alleen door agent gecontroleerd
                    agent_control_after = len(set(agent_valid_moves_after) - set(opponent_valid_moves_after)) # alleen door agent gecontroleerd

                    opponent_control_before = len(set(opponent_valid_moves_before) - set(agent_valid_moves_before)) # alleen door opponent gecontroleerd
                    opponent_control_after = len(set(opponent_valid_moves_after) - set(agent_valid_moves_after)) # alleen door oppponent gecontroleerd

                    mobility_reward = (0.1*(agent_control_after-agent_control_before) - 0.15*(opponent_control_after-opponent_control_before)) / normalization_factor

                    #mobility_reward = (0.05*(agent_valid_moves_after - agent_valid_moves_before))- (0.05*(opponent_valid_moves_after - opponent_valid_moves_before)) / normalization_factor

                    combined_reward += mobility_reward
                    episode_reward += combined_reward

                    # Sla transition op
                    current_board = state_obj.board.copy().astype(np.float32)
                    next_state = make_state(current_board, player=agent_player)
                    agent.store_transition(current_state, action, combined_reward, next_state, False)
                    #plot_board(state_obj.board, title=str((mobility_reward,combined_reward)), pause_time=1.5)
                    agent.update()
                    # if reward_opponent or reward_agent >= 0.1:
                    #     plot_board(state_obj.board, title= str((agent_player,action, opponent_action, reward_agent, reward_opponent)), pause_time=1000)

                else:
                    # Agent kan niet meer zetten, dus de tegenstander speelt door
                    cum_reward = 0.0
                    done_playout = False
                    while not done_playout:
                                    
                        # valid_moves = state_obj.get_all_moves()
                        # new_board = state_obj.board.copy().astype(np.float32)
                        # new_state = make_state(new_board, player=opponent_player)
                        # opp_action = opponent_agent.select_action(new_state, valid_moves, epsilon=0.1)
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

        # Einde episode: check of de agent won
        if final_agent_score > final_opponent_score:
            win_count += 1
        episodes_tracked += 1

        total_rewards += episode_reward
        rounds_in_interval += 1

        # Per win_rate_interval episodes, log stats
        if (episode + 1) % win_rate_interval == 0:
            win_rate = win_count / episodes_tracked
            win_rate_history.append(win_rate)
            avg_reward = total_rewards / rounds_in_interval
            avg_rewards_per_interval.append(avg_reward)

            # Sla model op als het de beste win_rate is
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

    # Plotten
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
