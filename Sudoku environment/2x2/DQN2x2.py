import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class SimpleCNNQNetwork(nn.Module):
    def __init__(self, input_shape=(4,4), num_actions=16):
        super(SimpleCNNQNetwork, self).__init__()
        # 3 kanalen: p1, p2, empty
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, batch_size=64, replay_size=1000, update_target_every=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.policy_net = SimpleCNNQNetwork().to(self.device)
        self.target_net = SimpleCNNQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_size)
        self.steps_done = 0

    def select_action(self, state, valid_actions, epsilon):
        # state shape: (2,4,4)
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,2,4,4)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze() # (16,)

        mask = np.ones(16) * (-1e9)
        for (r,c) in valid_actions:
            action_idx = r*4 + c
            mask[action_idx] = q_values[action_idx]

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            best_action_idx = np.argmax(mask)
            action = (best_action_idx // 4, best_action_idx % 4)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        action_idx = action[0] * 4 + action[1]  # Converteer (r, c) naar een enkele index
        self.memory.push(state, action_idx, reward, next_state, done)


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Converteer data naar tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)  # (B, 2, 4, 4)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)  # (B, 2, 4, 4)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # (B, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B,)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)  # (B,)

        # Bereken Q-waarden
        self.policy_net.train()
        q_values = self.policy_net(states_t)  # (B, 16)
        q_values = q_values.gather(1, actions_t).squeeze(1)  # (B,)

        # Bereken next Q-waarden
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states_t)  # (B, 16)
            next_actions = next_q_values_policy.argmax(dim=1).unsqueeze(1)  # (B, 1)
            next_q_values_target = self.target_net(next_states_t)  # (B, 16)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)  # (B,)
            next_q_values[dones_t] = 0.0  # Zet terminal states op 0

        # Doelberekening
        target = rewards_t + self.gamma * next_q_values
        loss = F.mse_loss(q_values, target)

        # Optimalisatie
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target netwerk
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

