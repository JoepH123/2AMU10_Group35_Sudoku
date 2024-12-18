# train_selfplay.py
import random
import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from env2x2 import DQLGameState  # Import your environment
from DQN2x2 import DQNAgent      # Import your DQN agent

def check_device():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU is NOT available. Using CPU.")

def save_model_as_pkl(agent, epsilon, filename="dqn_model.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "models")
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # If you want unique filenames each time, use filename_with_time
    # filename_with_time = filename.replace(".pkl", f"_{timestamp}.pkl")
    filepath = os.path.join(directory, filename)

    data_to_save = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }

    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model and parameters saved to {filepath}")

def make_state(board):
    # board shape: (4,4) with values in {-1,0,1}
    p1_channel = (board == 1).astype(np.float32)
    p2_channel = (board == -1).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    return np.stack([p1_channel, p2_channel, empty_channel], axis=0)  # (3,4,4)

def evaluate_agent(agent, env_class, episodes=100):
    # Evalueer agent tegen een vaste niet-lerende tegenstander (bvb random)
    wins = 0
    draws = 0
    for _ in range(episodes):
        env = env_class()
        env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                # Agent aan zet
                state = make_state(env.board)
                valid_moves = env.get_all_moves()
                if len(valid_moves) == 0:
                    # Check als spel over is
                    other_moves = env.get_all_moves(player=-1)
                    if len(other_moves) == 0:
                        done = True
                        break
                    else:
                        env.current_player = -env.current_player
                        continue
                action = agent.select_action(state, valid_moves, epsilon=0.0) # 0.0 voor pure greedy
                _, done_agent, info = env.step(action)
                done = done_agent
            else:
                # Random tegenstander
                valid_moves = env.get_all_moves()
                if len(valid_moves) == 0:
                    other_moves = env.get_all_moves(player=1)
                    if len(other_moves) == 0:
                        done = True
                        break
                    else:
                        env.current_player = -env.current_player
                        continue
                action = random.choice(valid_moves)
                _, done_opponent, info = env.step(action)
                done = done_opponent
            
        score_p1, score_p2 = env.score
        if score_p1 > score_p2:
            wins += 1
        elif score_p1 == score_p2:
            draws += 1
    win_rate = wins / episodes
    draw_rate = draws / episodes
    return win_rate, draw_rate


def main():
    check_device()

    num_episodes = 100_000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 50000  # Long decay for a large number of episodes
    epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay

    # Create two agents for self-play
    agent1 = DQNAgent(lr=1e-3, gamma=0.99, batch_size=64, replay_size=10000, update_target_every=1000)
    agent2 = DQNAgent(lr=1e-3, gamma=0.99, batch_size=64, replay_size=10000, update_target_every=1000)

    epsilon_agent1 = epsilon_start
    epsilon_agent2 = epsilon_start

    # Tracking performance
    win_rate_interval = 1000
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    episodes_tracked = 0

    agent1_win_rate_history = []
    agent2_win_rate_history = []
    draw_rate_history = []

    best_agent1_win_rate = 0.0
    best_agent2_win_rate = 0.0

    state_obj = DQLGameState()

    for episode in range(num_episodes):
        state_obj.reset()
        done = False

        # We keep track of scores throughout the episode
        # info["score"] = (score_p1, score_p2)
        # The current_player in state_obj determines which agent moves
        while not done:
            current_board = state_obj.board.copy().astype(np.float32)
            current_state = make_state(current_board)
            valid_moves = state_obj.get_all_moves()

            if len(valid_moves) == 0:
                # Current player cannot move
                # Check if the other player can move
                other_player = -state_obj.current_player
                other_moves = state_obj.get_all_moves(player=other_player)
                if len(other_moves) == 0:
                    # Both cannot move, game ends
                    done = True
                    break
                else:
                    # Switch player and continue
                    state_obj.current_player = other_player
                    continue

            # Choose the agent based on current_player
            if state_obj.current_player == 1:
                # Agent1's turn
                action = agent1.select_action(current_state, valid_moves, epsilon_agent1)
            else:
                # Agent2's turn
                action = agent2.select_action(current_state, valid_moves, epsilon_agent2)

            # Perform the step
            reward_agent, done_agent, info = state_obj.step(action)

            next_board = state_obj.board.copy().astype(np.float32)
            next_state = make_state(next_board)

            final_agent_score = info["score"][0]
            final_opponent_score = info["score"][1]

            # Determine the perspective-based reward
            if state_obj.current_player == 1:
                # Agent1 just played
                # reward = own_score - opp_score increment, but we can simplify:
                # If done_agent: the game ended after agent1's move
                if done_agent:
                    done = True
                    combined_reward = reward_agent
                    # Optionally add a final reward for win/lose:
                    # if final_agent_score > final_opponent_score:
                    #     combined_reward += 100.0
                    # elif final_agent_score < final_opponent_score:
                    #     combined_reward -= 100.0
                    agent1.store_transition(current_state, action, combined_reward, next_state, True)
                    agent1.update()
                else:
                    # Opponent (agent2) now moves
                    # Opponent action
                    opponent_moves = state_obj.get_all_moves()
                    if len(opponent_moves) == 0:
                        # Opponent can't move, check if game ends
                        other_player = -state_obj.current_player
                        other_moves = state_obj.get_all_moves(player=other_player)
                        if len(other_moves) == 0:
                            # Game ends
                            done = True
                            combined_reward = reward_agent
                            # if final_agent_score > final_opponent_score:
                            #     combined_reward += 100.0
                            # elif final_agent_score < final_opponent_score:
                            #     combined_reward -= 100.0
                            agent1.store_transition(current_state, action, combined_reward, next_state, True)
                            agent1.update()
                            break
                        else:
                            # Just switch player and continue
                            combined_reward = reward_agent
                            agent1.store_transition(current_state, action, combined_reward, next_state, False)
                            agent1.update()
                            state_obj.current_player = -state_obj.current_player
                            continue

                    # Opponent chooses a move
                    # agent2 is now at turn
                    opponent_action = agent2.select_action(next_state, opponent_moves, epsilon_agent2)
                    reward_opponent, done_opponent, info = state_obj.step(opponent_action)
                    done = done_opponent
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]

                    # Now agent1's combined reward for its step:
                    combined_reward = reward_agent - reward_opponent

                    # Store agent1 transition
                    agent1.store_transition(current_state, action, combined_reward, make_state(state_obj.board), done)
                    agent1.update()

                    if done:
                        # Game ended after opponent move
                        # No further store for agent2 yet (will be stored on their turn)
                        pass
                    else:
                        # Now we need to store the agent2 transition as well, because agent2 just played
                        # For agent2: perspective is reversed
                        # After agent2 move, agent2 sees reward as (p2_score - p1_score)
                        agent2_reward = reward_opponent - reward_agent  # symmetrical
                        agent2.store_transition(next_state, opponent_action, agent2_reward,
                                                make_state(state_obj.board), done)
                        agent2.update()

                    # Switch player back to agent1
                    state_obj.current_player = -state_obj.current_player

            else:
                # Agent2 just played (state_obj.current_player == -1)
                # Similar logic as above, but now from agent2's perspective
                if done_agent:
                    done = True
                    combined_reward = reward_agent
                    # if final_opponent_score > final_agent_score:
                    #     combined_reward += 100.0
                    # elif final_opponent_score < final_agent_score:
                    #     combined_reward -= 100.0
                    agent2.store_transition(current_state, action, combined_reward, next_state, True)
                    agent2.update()
                else:
                    # Opponent (agent1) now moves
                    opponent_moves = state_obj.get_all_moves()
                    if len(opponent_moves) == 0:
                        # Opponent can't move, check if game ends
                        other_player = -state_obj.current_player
                        other_moves = state_obj.get_all_moves(player=other_player)
                        if len(other_moves) == 0:
                            # Game ends
                            done = True
                            combined_reward = reward_agent
                            # if final_opponent_score > final_agent_score:
                            #     combined_reward += 100.0
                            # elif final_opponent_score < final_agent_score:
                            #     combined_reward -= 100.0
                            agent2.store_transition(current_state, action, combined_reward, next_state, True)
                            agent2.update()
                            break
                        else:
                            combined_reward = reward_agent
                            agent2.store_transition(current_state, action, combined_reward, next_state, False)
                            agent2.update()
                            state_obj.current_player = -state_obj.current_player
                            continue

                    # Opponent is agent1
                    opponent_action = agent1.select_action(next_state, opponent_moves, epsilon_agent1)
                    reward_opponent, done_opponent, info = state_obj.step(opponent_action)
                    done = done_opponent
                    final_agent_score = info["score"][0]
                    final_opponent_score = info["score"][1]

                    combined_reward = reward_agent - reward_opponent
                    agent2.store_transition(current_state, action, combined_reward,
                                            make_state(state_obj.board), done)
                    agent2.update()

                    if done:
                        pass
                    else:
                        # Agent1's perspective
                        agent1_reward = reward_opponent - reward_agent
                        agent1.store_transition(next_state, opponent_action, agent1_reward,
                                                make_state(state_obj.board), done)
                        agent1.update()

                    state_obj.current_player = -state_obj.current_player

        # Epsilon decay
        if epsilon_agent1 > epsilon_end:
            epsilon_agent1 -= epsilon_step
        if epsilon_agent2 > epsilon_end:
            epsilon_agent2 -= epsilon_step

        # Determine winner for stats
        # After the game ends:
        score_p1, score_p2 = state_obj.score
        if score_p1 > score_p2:
            agent1_wins += 1
        elif score_p2 > score_p1:
            agent2_wins += 1
        else:
            draws += 1
        episodes_tracked += 1

        # Log win rates at intervals
        if (episode + 1) % win_rate_interval == 0:
            agent1_eval_win_rate, agent1_eval_draw_rate = evaluate_agent(agent1, DQLGameState)
            agent2_eval_win_rate, agent2_eval_draw_rate = evaluate_agent(agent2, DQLGameState)
            print(f"Agent1 eval vs random: Win {agent1_eval_win_rate:.2f}, Draw {agent1_eval_draw_rate:.2f}")
            print(f"Agent2 eval vs random: Win {agent2_eval_win_rate:.2f}, Draw {agent2_eval_draw_rate:.2f}")

            agent1_win_rate = agent1_wins / episodes_tracked
            agent2_win_rate = agent2_wins / episodes_tracked
            draw_rate = draws / episodes_tracked

            agent1_win_rate_history.append(agent1_win_rate)
            agent2_win_rate_history.append(agent2_win_rate)
            draw_rate_history.append(draw_rate)

            print(f"Episode {episode+1}/{num_episodes}")
            print(f"Agent1 Win Rate: {agent1_win_rate:.2f}")
            print(f"Agent2 Win Rate: {agent2_win_rate:.2f}")
            print(f"Draw Rate: {draw_rate:.2f}")
            print(f"Epsilon Agent1: {epsilon_agent1:.3f}, Epsilon Agent2: {epsilon_agent2:.3f}")

            # Save best models
            if agent1_win_rate > best_agent1_win_rate:
                best_agent1_win_rate = agent1_win_rate
                save_model_as_pkl(agent1, epsilon_agent1, filename="best_dqn_model_agent1.pkl")

            if agent2_win_rate > best_agent2_win_rate:
                best_agent2_win_rate = agent2_win_rate
                save_model_as_pkl(agent2, epsilon_agent2, filename="best_dqn_model_agent2.pkl")

            # Reset counters
            agent1_wins = 0
            agent2_wins = 0
            draws = 0
            episodes_tracked = 0

    # Final save
    save_model_as_pkl(agent1, epsilon_agent1, filename="final_dqn_model_agent1.pkl")
    save_model_as_pkl(agent2, epsilon_agent2, filename="final_dqn_model_agent2.pkl")

    # Plot results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    x_axis = range(win_rate_interval, num_episodes + 1, win_rate_interval)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, agent1_win_rate_history, label="Agent1 Win Rate")
    plt.plot(x_axis, agent2_win_rate_history, label="Agent2 Win Rate")
    plt.plot(x_axis, draw_rate_history, label="Draw Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Rate")
    plt.title("Win/Draw Rate Over Training")
    plt.legend()
    plt.grid()
    win_rate_plot_path = os.path.join(plots_dir, "win_draw_rate_plot.png")
    plt.savefig(win_rate_plot_path)
    plt.close()
    print(f"Win/draw rate plot saved to {win_rate_plot_path}")

if __name__ == "__main__":
    main()
