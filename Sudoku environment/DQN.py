# dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape=(9,9), num_actions=81):
        super(CNNQNetwork, self).__init__()
        # input_shape = (9,9) single channel (can reshape to 1x9x9)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # After conv layers: shape still (64,9,9)
        # Flatten
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 128),  # Minder neuronen
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # x expected shape: (batch_size, 1, 9, 9)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),  # Compact formaat
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return zip(*[self.memory[i] for i in indices])


    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, batch_size=128, replay_size=10000, update_target_every=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.policy_net = CNNQNetwork().to(self.device)
        self.target_net = CNNQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_size)
        self.steps_done = 0

    def select_action(self, state, valid_actions, epsilon):
        # state shape: (9,9)
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            # shape: (1,1,9,9)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze() # shape (81,)
        
        # Mask invalid actions with large negative number so they won't be chosen
        mask = np.ones(81) * (-1e9)
        for (r,c) in valid_actions:
            action_idx = r*9 + c
            mask[action_idx] = q_values[action_idx]

        if random.random() < epsilon:
            # random action from valid actions
            action = random.choice(valid_actions)
        else:
            # argmax from mask
            best_action_idx = np.argmax(mask)
            action = (best_action_idx // 9, best_action_idx % 9)

        return action
    
    def store_transition(self, state, action, reward, next_state, done):

        # Convert action (r, c) to a single int index
        action_idx = action[0] * 9 + action[1]
        self.memory.push(state, action_idx, reward, next_state, done)


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states_np = np.array(states)  
        next_states_np = np.array(next_states)  
        states_t = torch.from_numpy(states_np).float().to(self.device).unsqueeze(1).to(self.device)  # Shape: (B, 1, 9, 9)
        next_states_t = torch.from_numpy(next_states_np).float().to(self.device).unsqueeze(1)  # Shape: (B, 1, 9, 9)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Current Q values
        self.policy_net.train()
        q_values = self.policy_net(states_t) # (B,81)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1) # Q(s,a)

        # Double Q-learning:
        # 1) From policy net, select best action in next_state
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states_t) # (B,81)
            next_actions = next_q_values_policy.argmax(dim=1) # best actions according to policy net

            # 2) Evaluate these actions in target net
            next_q_values_target = self.target_net(next_states_t)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # If done, no future reward
            next_q_values[dones_t] = 0.0

        # Target
        target = rewards_t + self.gamma * next_q_values

        # Loss
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


