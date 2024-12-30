# train_6x6.py

import random
import numpy as np
import time
import os
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from env import DQLGameState6x6
from DQN import DQNAgent
from test import plot_board

def check_device():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is NOT available. Using CPU.")

def save_model_as_pkl(agent, epsilon, filename="dqn_6x6_model.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models_6x6")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(models_dir, filename)

    data = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved to {filepath}")

def make_state(board, player=1):
    """
    board shape: (6,6), values in {-1, 0, +1}.
    Maak 3 kanalen:
      kanaal 0: cellen == +player
      kanaal 1: cellen == -player
      kanaal 2: cellen == 0
    """
    p1_channel = (board == player).astype(np.float32)
    p2_channel = (board == -player).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,6,6)

def random_opponent_move(env, opponent_player):
    valid_moves = env.get_all_moves(player=opponent_player)
    if len(valid_moves) == 0:
        return None
    return random.choice(valid_moves)

def main():
    check_device()

    num_episodes = 80_000
    agent = DQNAgent(
        lr=1e-3,        # iets hogere lr
        gamma=0.99,
        batch_size=64,
        replay_size=100000,
        update_target_every=250,
        tau=0.005
    )

    # Epsilon decay
    epsilon_start = 0.6
    epsilon_mid = 0.1
    epsilon_end = 0.01
    eps_phase1_steps = 40_000
    eps_phase2_steps = 40_000
    total_decay = eps_phase1_steps + eps_phase2_steps
    eps_step_phase1 = (epsilon_start - epsilon_mid) / eps_phase1_steps
    eps_step_phase2 = (epsilon_mid - epsilon_end) / eps_phase2_steps

    # Stats
    win_interval = 500
    win_count = 0
    episodes_count = 0
    best_win_rate = 0.0
    winrate_list = []
    intervals_list = []

    env = DQLGameState6x6()

    for episode in tqdm(range(num_episodes), desc="Training 6x6 DQN", unit="episode"):
        # Bepaal epsilon
        if episode < eps_phase1_steps:
            epsilon = max(epsilon_mid, epsilon_start - episode * eps_step_phase1)
        elif episode < total_decay:
            steps_in_phase2 = episode - eps_phase1_steps
            epsilon = max(epsilon_end, epsilon_mid - steps_in_phase2 * eps_step_phase2)
        else:
            epsilon = epsilon_end

        # Random bepalen of agent player=1 of player=-1 is
        if random.random() < 0.5:
            agent_player = 1
        else:
            agent_player = -1
        opp_player = -agent_player

        env.reset()
        plot_board(env.board, title= 'agent is: '+str(agent_player), pause_time=2)

        # Als de opponent start
        if env.current_player == opp_player:
            # Opponent random move
            opp_action = random_opponent_move(env, opp_player)
            if opp_action is not None:
                rew, done, info = env.step(opp_action)

        done = False
        agent_score = 0
        opp_score = 0
        episode_reward = 0.0

        while not done:
            if env.current_player == agent_player:
                valid_moves = env.get_all_moves(player=agent_player)
                if len(valid_moves) == 0:
                    # Kan niet zetten
                    env.player_can_move[agent_player] = False
                    # Check of opponent kan
                    opp_moves = env.get_all_moves(player=opp_player)
                    if len(opp_moves) == 0:
                        done = True
                    else:
                        # Wissel naar opponent
                        env.current_player = opp_player
                    continue
                # Anders wel moves
                curr_board = env.board.copy()
                curr_state = make_state(curr_board, player=agent_player)

                action = agent.select_action(curr_state, valid_moves, epsilon)
                reward, done_step, info = env.step(action)
                plot_board(env.board, title= 'reward is: '+str(reward), pause_time=2)
                # done_step => check of direct game end

                next_board = env.board.copy()
                next_state = make_state(next_board, player=agent_player)
                agent.store_transition(curr_state, action, reward, next_state, done_step)
                agent.update()

                episode_reward += reward
                agent_score, opp_score = info["score"]

                if done_step:
                    done = True
                else:
                    # check next player
                    pass
            else:
                # Opponent's turn (random)
                opp_moves = env.get_all_moves(player=opp_player)
                if len(opp_moves) == 0:
                    env.player_can_move[opp_player] = False
                    # check agent moves
                    agent_moves = env.get_all_moves(player=agent_player)
                    if len(agent_moves) == 0:
                        done = True
                    else:
                        env.current_player = agent_player
                else:
                    opp_action = random_opponent_move(env, opp_player)
                    if opp_action is not None:
                        rew_opp, done_opp, info_opp = env.step(opp_action)
                        plot_board(env.board, title= 'reward opponent is: '+str(rew_opp), pause_time=2)
                        agent_score, opp_score = info_opp["score"]
                        if done_opp:
                            done = True
                    else:
                        # No action
                        done = True
            # while loop continue

        # Einde episode
        if agent_score > opp_score:
            win_count += 1
        episodes_count += 1

        # Log om de 500 episodes
        if (episode+1) % win_interval == 0:
            win_rate = win_count / episodes_count
            print(f"Episode {episode+1}/{num_episodes}, Eps={epsilon:.3f}, WinRate(last {win_interval})={win_rate:.2f}")
            intervals_list.append(episode+1)
            winrate_list.append(win_rate)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_model_as_pkl(agent, epsilon, filename="dqn_6x6_best_model.pkl")

            # reset for next interval
            win_count = 0
            episodes_count = 0

    # Opslaan model
    save_model_as_pkl(agent, epsilon, filename="dqn_6x6_final_model.pkl")

    # Plot winrate
    plt.plot(intervals_list, winrate_list, label="Win Rate")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("6x6 DQN Training")
    plt.grid()
    plt.legend()
    plt.savefig("winrate_6x6.png")
    plt.close()
    print("Done training 6x6 DQN.")


if __name__ == "__main__":
    main()
