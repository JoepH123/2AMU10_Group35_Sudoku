# train.py
import random
import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from env import DQLGameState
from DQN import DQNAgent


def greedy_opponent_move(state):
    moves = state.get_all_moves()
    
    if len(moves) == 0:
        return None  

    best_move = None
    max_reward = float('-inf')

    for move in moves:
        state_copy = state.clone()
        
        reward, _, _ = state_copy.step(move)
        
        if reward > max_reward:
            max_reward = reward
            best_move = move

    return best_move


def random_opponent_move(state):
    # state is a DQLGameState
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
    """Save the policy network and metadata as a pickle file in the 'models' directory."""
    # Dynamisch pad naar de map 'models'
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Map van het script
    directory = os.path.join(script_dir, "models")

    # Zorg dat de map bestaat
    os.makedirs(directory, exist_ok=True)

    # Volledig pad naar het bestand
    filepath = os.path.join(directory, filename)

    # Maak een dictionary van de data om op te slaan
    data_to_save = {
        "policy_state_dict": agent.policy_net.state_dict(),
        "epsilon": epsilon
    }

    # Sla de data op met pickle
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model and parameters saved to {filepath}")

def main():
    check_device()
    num_episodes = 50_000
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 10_000 # decay steps
    agent = DQNAgent(lr=1e-3, gamma=0.99, batch_size=128, replay_size=10_000, update_target_every=1_000)

    epsilon = epsilon_start
    epsilon_step = (epsilon_start - epsilon_end)/epsilon_decay

    # For tracking win rate every 100 episodes
    win_rate_history = []
    win_count = 0
    episodes_tracked = 0
    win_rate_interval = 100
    rewards_history = []
    total_rewards_history = []

    state_obj = DQLGameState()

    for episode in range(num_episodes):
        state_obj.reset()
        done = False
        steps = 0
        total_rewards = 0

        # Keep track of final scores for win calculation
        # The agent is Player 1, opponent is Player 2
        final_agent_score = 0
        final_opponent_score = 0

        while not done:
            # Agent's turn
            valid_moves = state_obj.get_all_moves()
            if len(valid_moves) == 0:
                # Agent cannot move
                # Check if other player can move
                other_player = -state_obj.current_player
                other_moves = state_obj.get_all_moves(player=other_player)
                if len(other_moves) == 0:
                    # Both can't move
                    done = True
                    break
                else:
                    # Switch player
                    state_obj.current_player = other_player
                    continue

            current_state = state_obj.board.copy().astype(np.float32)
            action = agent.select_action(current_state, valid_moves, epsilon)
            reward_agent, done_agent, info = state_obj.step(action)

            # Update agent's final score
            final_agent_score = info["score"][0]
            final_opponent_score = info["score"][1]

            if done_agent:
                # No opponent move if done
                reward_opponent = 0
                done = True
                # combined reward
                combined_reward = reward_agent - reward_opponent
                agent.store_transition(current_state, action, combined_reward, state_obj.board.copy().astype(np.float32), True)
                agent.update()
                break

            # Opponent's turn (random)
            opponent_action = random_opponent_move(state_obj)
            if opponent_action is None:
                # Opponent can't move
                other_player = -state_obj.current_player
                other_moves = state_obj.get_all_moves(player=other_player)
                if len(other_moves) == 0:
                    # Both can't move
                    done = True
                    # No additional reward from opponent
                    combined_reward = reward_agent - 0
                    agent.store_transition(current_state, action, combined_reward, state_obj.board.copy().astype(np.float32), True)
                    agent.update()
                    break
                else:
                    # Switch and continue
                    state_obj.current_player = other_player
                    # combined reward after just the agent's move
                    combined_reward = reward_agent - 0
                    agent.store_transition(current_state, action, combined_reward, state_obj.board.copy().astype(np.float32), False)
                    agent.update()
                    continue
            
            reward_opponent, done_opponent, info = state_obj.step(opponent_action)
            done = done_opponent

            # Update scores
            final_agent_score = info["score"][0]
            final_opponent_score = info["score"][1]

            # combined reward for the agent
            combined_reward = reward_agent - reward_opponent
            total_rewards += combined_reward

            # Store in replay memory
            next_state = state_obj.board.copy().astype(np.float32)
            agent.store_transition(current_state, action, combined_reward, next_state, done)
            agent.update()

            steps += 1

        rewards_history.append(total_rewards)

        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon -= epsilon_step

        # Track win/loss
        # Win if agent_score > opponent_score
        if final_agent_score > final_opponent_score:
            win_count += 1
        episodes_tracked += 1

        # Print win rate every 1000 episodes
        if (episode + 1) % win_rate_interval == 0:
            win_rate = win_count / episodes_tracked
            win_rate_history.append(win_rate)
            total_rewards_history.append(sum(rewards_history)/win_rate_interval)
            
            print( 'Average reward for last 100 games:',sum(rewards_history)/win_rate_interval)
            print(f"Episode {episode+1}/{num_episodes} completed. Epsilon: {epsilon:.3f}")
            print(f"Win Rate over last {win_rate_interval} episodes: {win_rate:.2f}%")
            print(f"Last Episode Score: {state_obj.score}")
            # Reset counters
            win_count = 0
            episodes_tracked = 0
            rewards_history = []

    print("Training completed.")
    save_model_as_pkl(agent, epsilon)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Pad naar het script
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(win_rate_interval, num_episodes + 1, win_rate_interval), win_rate_history, label="Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Training")
    plt.legend()
    plt.grid()

    # Sla de plot op in de 'plots'-map
    plot_path = os.path.join(plots_dir, "win_rate_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Win rate plot saved to {plot_path}")

if __name__ == "__main__":
    main()
