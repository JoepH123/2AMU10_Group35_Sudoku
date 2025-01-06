import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


def generate_transformed_transitions(state, next_state, action):
    """
    Generates different transformations (identity, 180° rotation, horizontal flip, vertical flip)
    of a given transition (state, next_state, action). This is a form of data augmentation for training.
    
    Parameters:
    -----------
    state : np.ndarray
        Current state with shape (3, 9, 9), representing the board channels.
    next_state : np.ndarray
        Next state with the same shape as 'state'.
    action : tuple
        A tuple (row, col) representing the chosen action on the board.
    
    Returns:
    --------
    transformations : list of tuples
        Each tuple has the transformed state, next_state, and the transformed action.
        The transformations include the original (identity), 180° rotation, horizontal flip, and vertical flip.
    """
    transformations = []
    (row, col) = action

    # Identity
    transformations.append((state, next_state, (row, col)))

    # 180° Rotation
    rotated_state_180 = np.array([np.rot90(channel, 2) for channel in state])
    rotated_next_state_180 = np.array([np.rot90(channel, 2) for channel in next_state])
    r_a_180, c_a_180 = 8 - row, 8 - col
    transformations.append((rotated_state_180, rotated_next_state_180, (r_a_180, c_a_180)))

    # Horizontal flip
    flipped_state_h = np.array([np.flip(channel, axis=1) for channel in state])
    flipped_next_state_h = np.array([np.flip(channel, axis=1) for channel in next_state])
    r_a_h, c_a_h = row, 8 - col
    transformations.append((flipped_state_h, flipped_next_state_h, (r_a_h, c_a_h)))

    # Vertical flip
    flipped_state_v = np.array([np.flip(channel, axis=0) for channel in state])
    flipped_next_state_v = np.array([np.flip(channel, axis=0) for channel in next_state])
    r_a_v, c_a_v = 8 - row, col
    transformations.append((flipped_state_v, flipped_next_state_v, (r_a_v, c_a_v)))

    return transformations


class CNNQNetwork(nn.Module):
    """
    A Convolutional Neural Network for Q-value estimation (DQN).
    Expects a (3, 9, 9) input (three channels, 9x9 board).
    Outputs Q-values for 81 discrete actions (9x9 possible moves).
    """
    def __init__(self, input_shape=(9,9), num_actions=81):
        super(CNNQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Flatten: 64 * 9 * 9 = 5184
        self.fc = nn.Sequential(
            nn.Linear(5184, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        """
        Forward pass of the CNN. 
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 9, 9).
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, 81), representing the Q-values for each possible action.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ReplayMemory:
    """
    A simple replay buffer to store and retrieve transitions for training.
    """
    def __init__(self, capacity):
        """
        Initializes a replay memory with a maximum capacity.
        
        Parameters:
        -----------
        capacity : int
            Maximum number of stored transitions.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        
        Parameters:
        -----------
        state : np.ndarray
        action : int
        reward : float
        next_state : np.ndarray
        done : bool
        """
        self.memory.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the replay buffer.
        
        Parameters:
        -----------
        batch_size : int
            Number of transitions to sample.
        
        Returns:
        --------
        tuple
            A tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return zip(*[self.memory[i] for i in indices])

    def __len__(self):
        """
        Returns the current number of transitions in memory.
        """
        return len(self.memory)


class DQNAgent:
    """
    A DQN agent that uses a CNN-based Q-network to select and train on 9x9 board actions.
    Implements Double DQN logic, replay memory, and target network updates.
    """
    def __init__(
        self,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        replay_size=100000,
        update_target_every=250,
        tau=0.005
    ):
        """
        Initializes the DQN agent.
        
        Parameters:
        -----------
        lr : float
            Learning rate for the optimizer.
        gamma : float
            Discount factor for future rewards.
        batch_size : int
            Batch size for training.
        replay_size : int
            Capacity of the replay buffer.
        update_target_every : int
            Frequency (in steps) to perform a hard update of the target network.
        tau : float
            Coefficient for soft updates; if unused, the agent falls back to hard updates.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.tau = tau  # soft update parameter

        self.policy_net = CNNQNetwork().to(self.device)
        self.target_net = CNNQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_size)
        self.steps_done = 0

    def select_action(self, state, valid_actions, epsilon):
        """
        Chooses an action based on an epsilon-greedy policy over the Q-values.
        
        Parameters:
        -----------
        state : np.ndarray
            The current state, shape (3, 9, 9).
        valid_actions : list of tuples
            List of (row, col) tuples representing valid moves.
        epsilon : float
            Epsilon value for epsilon-greedy exploration.
        
        Returns:
        --------
        tuple
            A (row, col) action selected from valid_actions.
        """
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3,9,9)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze()  # (81,)

        mask = np.ones(81) * (-1e9)
        for (r, c) in valid_actions:
            action_idx = r * 9 + c
            mask[action_idx] = q_values[action_idx]

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            best_action_idx = np.argmax(mask)
            action = (best_action_idx // 9, best_action_idx % 9)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer. Converts the (row, col) action into
        an integer index for the Q-network (row * 9 + col).
        
        Parameters:
        -----------
        state : np.ndarray
        action : tuple
        reward : float
        next_state : np.ndarray
        done : bool
        """
        action_idx = action[0] * 9 + action[1]
        self.memory.push(state, action_idx, reward, next_state, done)

    def soft_update(self):
        """
        Performs a soft update of the target network:
        target_param = tau * policy_param + (1 - tau) * target_param
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self):
        """
        Performs one step of the training procedure:
          - Samples a batch of experiences from the replay buffer.
          - Computes the Q-values and target Q-values (Double DQN).
          - Calculates and backpropagates the loss (MSE or Smooth L1).
          - Optionally performs a soft or hard update of the target network.
        """

        # Ensure we have enough samples in the replay buffer
        if len(self.memory) < self.batch_size:
            return

        # Retrieve a random batch from replay
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert numpy arrays to Torch tensors
        states_np = np.stack(states)
        next_states_np = np.stack(next_states)

        states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Switch to training mode and compute Q-values for current states
        self.policy_net.train()
        q_values = self.policy_net(states_t)  # shape: (B, 81)
        q_values = q_values.gather(1, actions_t).squeeze(1)  # shape: (B,)

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Actions chosen by policy_net
            next_q_policy = self.policy_net(next_states_t)  # (B, 81)
            next_actions = next_q_policy.argmax(dim=1).unsqueeze(1)  # (B,1)

            # Q-values of these actions from target_net
            next_q_target = self.target_net(next_states_t)  # (B, 81)
            next_q_target = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)
            next_q_target[dones_t] = 0.0  # Zero Q-value for terminal states

        # DQN target
        target = rewards_t + self.gamma * next_q_target

        # Compute MSE loss 
        loss = F.mse_loss(q_values, target)

        # Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update step count
        self.steps_done += 1

        # Soft update call (commented out by default)
        # self.soft_update()

        # Hard update
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
