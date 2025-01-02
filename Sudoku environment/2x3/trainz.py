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

def load_model(agent, filename="dqn_model.pkl"):
    """Laad een opgeslagen model en initialiseer de agent met de gewichten."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models_6x6")
    file_path = os.path.join(models_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file '{file_path}' not found.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    agent.policy_net.load_state_dict(data["policy_state_dict"])
    agent.target_net.load_state_dict(data["policy_state_dict"])  # Synchroniseer target_net
    print(f"Model loaded from {file_path}.")

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
    Maakt 3 kanalen uit het 6x6 bord:
     - Kanaal0: cellen == +player
     - Kanaal1: cellen == -player
     - Kanaal2: cellen == 0
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
        lr=1e-3,           # Leer wat sneller
        gamma=0.99,
        batch_size=128,
        replay_size=5000,
        update_target_every=500,
        tau=0.005
    )

    # Epsilon: start hoger (0.8) -> mid (0.3) -> end (0.05)
    eps_start = 0.5
    eps_mid = 0.3
    eps_end = 0.05
    eps_phase1 = 20_000
    eps_phase2 = 30_000
    total_decay = eps_phase1 + eps_phase2
    step_phase1 = (eps_start - eps_mid) / eps_phase1
    step_phase2 = (eps_mid - eps_end) / eps_phase2

    # Stats
    win_interval = 500
    win_count = 0
    episodes_count = 0
    best_win_rate = 0.0
    winrate_list = []
    intervals_list = []

    # Extra
    normalization_factor = 7.0
    mobility_coeff = 0 #0.05

    # Probeer een bestaand model te laden
    model_filename = "dqn_6x6_random_2_model.pkl"  # Pas aan naar jouw modelbestand
    try:
        load_model(agent, filename=model_filename)
        print("Starting training with preloaded model.")
    except FileNotFoundError:
        print("No pretrained model found. Starting training from scratch.")


    env = DQLGameState6x6()

    for episode in tqdm(range(num_episodes), desc="Training 6x6 with cumReward+mobility", unit="episode"):
        # Bepaal epsilon
        if episode < eps_phase1:
            epsilon = max(eps_mid, eps_start - episode * step_phase1)
        elif episode < total_decay:
            steps_in_phase2 = episode - eps_phase1
            epsilon = max(eps_end, eps_mid - steps_in_phase2 * step_phase2)
        else:
            epsilon = eps_end

        # Agent or Opp first
        if random.random() < 0.5:
            agent_player = 1
        else:
            agent_player = -1
        opp_player = -agent_player

        env.reset()
        #plot_board(env.board, title=f"Episode {episode}: Start (Agent={agent_player})", pause_time=1.5)

        # If opp starts
        if env.current_player == opp_player:
            opp_action = random_opponent_move(env, opp_player)
            if opp_action is not None:
                opp_reward, done_opp, info_opp = env.step(opp_action)
                #plot_board(env.board, title=f"Opponent first move rew={opp_reward}", pause_time=1.5)

        done = False
        agent_score = 0
        opp_score = 0

        while not done:
            if env.current_player == agent_player:
                valid_moves = env.get_all_moves(player=agent_player)
                if len(valid_moves) == 0:
                    env.player_can_move[agent_player] = False
                    # Check opponent
                    opp_moves = env.get_all_moves(player=opp_player)
                    if len(opp_moves) == 0:
                        done = True
                    else:
                        env.current_player = opp_player
                    continue

                # Mobility BEFORE
                agent_moves_bef = len(valid_moves)
                opp_moves_bef = len(env.get_all_moves(player=opp_player))

                curr_board = env.board.copy()
                curr_state = make_state(curr_board, player=agent_player)

                action = agent.select_action(curr_state, valid_moves, epsilon)
                agent_reward, doneA, infoA = env.step(action)
                #plot_board(env.board, title=f"Agent move rew={agent_reward}", pause_time=1.5)

                agent_score, opp_score = infoA["score"]

                if doneA:
                    # Mobility na agentmove
                    agent_moves_after = len(env.get_all_moves(player=agent_player))
                    opp_moves_after = len(env.get_all_moves(player=opp_player))
                    mob_rw = mobility_coeff * ((agent_moves_after - agent_moves_bef)
                                               - (opp_moves_after - opp_moves_bef))
                    combined_reward = (agent_reward + mob_rw) / normalization_factor

                    next_state = make_state(env.board.copy(), player=agent_player)
                    agent.store_transition(curr_state, action, combined_reward, next_state, True)
                    agent.update()
                    done = True
                else:
                    # Opponent tries
                    opp_action = random_opponent_move(env, opp_player)
                    if opp_action is None:
                        # Opponent can't move => no oppReward
                        agent_moves_after = len(env.get_all_moves(player=agent_player))
                        opp_moves_after = len(env.get_all_moves(player=opp_player))
                        mob_rw = mobility_coeff * ((agent_moves_after - agent_moves_bef)
                                                   - (opp_moves_after - opp_moves_bef))
                        combined_reward = (agent_reward) / normalization_factor # no mobility reward, opp has no mobility at all

                        next_state = make_state(env.board.copy(), player=agent_player)
                        agent.store_transition(curr_state, action, combined_reward, next_state, False)
                        #plot_board(env.board, title=f"Agent continues rew={combined_reward}", pause_time=5)
                        agent.update()
                        if env.is_terminal():
                            done = True
                    else:
                        opp_rew, done_opp, info_opp = env.step(opp_action)
                        #plot_board(env.board, title=f"Opponent rew={opp_rew}", pause_time=1.5)
                        agent_score, opp_score = info_opp["score"]

                        # Mobility na agent + 1 opp move
                        agent_moves_after = len(env.get_all_moves(player=agent_player))
                        opp_moves_after = len(env.get_all_moves(player=opp_player))
                        mob_rw = mobility_coeff * ((agent_moves_after - agent_moves_bef)
                                                   - (opp_moves_after - opp_moves_bef))

                        combined_reward = (agent_reward - opp_rew + mob_rw)

                        # Check of agent nu KAN
                        agent_moves_now = env.get_all_moves(player=agent_player)
                        if len(agent_moves_now) == 0 and not done_opp:
                            # Agent kan niet meer, maar spel is niet done => Opponent kan doorspelen
                            extra_opp_rew = 0.0
                            done_extra = done_opp
                            while True:
                                next_opp_moves = env.get_all_moves(player=opp_player)
                                if not next_opp_moves:
                                    # nu is t mischien done
                                    if env.is_terminal():
                                        done_extra = True
                                    break
                                next_opp_action = random_opponent_move(env, opp_player)
                                if next_opp_action is None:
                                    if env.is_terminal():
                                        done_extra = True
                                    break
                                r_opp2, done2, info2 = env.step(next_opp_action)
                                extra_opp_rew += r_opp2
                                if done2:
                                    done_extra = True
                                    break

                            combined_reward = (combined_reward - extra_opp_rew) / normalization_factor
                            next_state = make_state(env.board.copy(), player=agent_player)
                            agent.store_transition(curr_state, action, combined_reward, next_state, True)
                            #plot_board(env.board, title=f"Opp continues rew={combined_reward}", pause_time=5)
                            agent.update()
                            done = True
                        else:
                            # Normaal scenario
                            combined_reward /= normalization_factor
                            next_state = make_state(env.board.copy(), player=agent_player)
                            agent.store_transition(curr_state, action, combined_reward, next_state, done_opp)
                            #plot_board(env.board, title=f"Combined reward={combined_reward}", pause_time=2)
                            agent.update()
                            if done_opp:
                                done = True
            else:
                # Opponent turn
                opp_moves = env.get_all_moves(player=opp_player)
                if not opp_moves:
                    env.player_can_move[opp_player] = False
                    agent_moves_2 = env.get_all_moves(player=agent_player)
                    if not agent_moves_2:
                        done = True
                    else:
                        env.current_player = agent_player
                else:
                    opp_a = random.choice(opp_moves)
                    opp_rew, done_op, info_op = env.step(opp_a)
                    #plot_board(env.board, title=f"Opp turn rew={opp_rew}", pause_time=1.5)
                    if done_op:
                        done = True

    # Einde training
        agent_score = env.score[0] if agent_player == 1 else env.score[1]
        opp_score = env.score[1] if agent_player == 1 else env.score[0]

        if agent_score > opp_score:
            win_count += 1
        episodes_count += 1

        if (episode+1) % win_interval == 0:
            w_rate = win_count / episodes_count
            print(f"[Ep {episode+1}/{num_episodes}] Eps={epsilon:.3f}, WinRate={w_rate:.2f}")
            intervals_list.append(episode+1)
            winrate_list.append(w_rate)

            if w_rate > best_win_rate:
                best_win_rate = w_rate
                save_model_as_pkl(agent, epsilon, filename="dqn_6x6_best_model.pkl")

            # Reset counters
            win_count = 0
            episodes_count = 0

    # Save final
    save_model_as_pkl(agent, epsilon, filename="dqn_6x6_final_model.pkl")

    # Plot
    plt.plot(intervals_list, winrate_list, label="Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("6x6 DQN Training - Mobility + Cumulative Opp Reward")
    plt.grid()
    plt.legend()
    plt.savefig("winrate_6x6.png")
    plt.close()
    print("Done training 6x6 with cumulative negative reward + mobility reward.")


if __name__ == "__main__":
    main()
