# DQN_6x6.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

def generate_transformed_transitions(state, next_state, action):
    """Data augmentation: identity, 180-rotation, horizontal flip, vertical flip."""
    transformations = []
    (row, col) = action

    # Identity
    transformations.append((state, next_state, (row, col)))

    # 180° rotation
    rot180_state = np.array([np.rot90(channel, 2) for channel in state])
    rot180_next_state = np.array([np.rot90(channel, 2) for channel in next_state])
    r_a_180, c_a_180 = 5 - row, 5 - col  # 6x6 => indices 0..5
    transformations.append((rot180_state, rot180_next_state, (r_a_180, c_a_180)))

    # Horizontal flip
    flip_h_state = np.array([np.flip(channel, axis=1) for channel in state])
    flip_h_next_state = np.array([np.flip(channel, axis=1) for channel in next_state])
    r_a_h, c_a_h = row, 5 - col
    transformations.append((flip_h_state, flip_h_next_state, (r_a_h, c_a_h)))

    # Vertical flip
    flip_v_state = np.array([np.flip(channel, axis=0) for channel in state])
    flip_v_next_state = np.array([np.flip(channel, axis=0) for channel in next_state])
    r_a_v, c_a_v = 5 - row, col
    transformations.append((flip_v_state, flip_v_next_state, (r_a_v, c_a_v)))

    return transformations

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape=(6,6), num_actions=36):
        super(CNNQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Flatten => 64 * 6 * 6 = 2304
        self.fc = nn.Sequential(
            nn.Linear(2304, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        # x shape: (B, 3, 6, 6)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        transitions = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, lr=1e-4, gamma=0.99, batch_size=64,
                 replay_size=100000, update_target_every=250, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.tau = tau

        self.policy_net = CNNQNetwork(input_shape=(6,6), num_actions=36).to(self.device)
        self.target_net = CNNQNetwork(input_shape=(6,6), num_actions=36).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_size)
        self.steps_done = 0

    def select_action(self, state, valid_actions, epsilon):
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # state shape => (1,3,6,6)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze()  # (36,)

        mask = np.ones(36) * -1e9
        for (r, c) in valid_actions:
            idx = r * 6 + c
            mask[idx] = q_values[idx]

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            best_idx = np.argmax(mask)
            action = (best_idx // 6, best_idx % 6)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # Data augmentatie
        transitions = generate_transformed_transitions(state, next_state, action)
        for s_aug, s_next_aug, a_aug in transitions:
            a_idx = a_aug[0]*6 + a_aug[1]
            self.memory.push(s_aug, a_idx, reward, s_next_aug, done)

        # Extra opslaan als reward groot is
        if abs(reward) > 0.1:
            for _ in range(3):
                for s_aug, s_next_aug, a_aug in transitions:
                    a_idx = a_aug[0]*6 + a_aug[1]
                    self.memory.push(s_aug, a_idx, reward, s_next_aug, done)

    def soft_update(self):
        # tau * policy + (1-tau)*target
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self):
        # Check replay
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_np = np.stack(states)
        next_states_np = np.stack(next_states)

        states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Q(s,a)
        self.policy_net.train()
        q_values = self.policy_net(states_t)  # (B,36)
        q_values = q_values.gather(1, actions_t).squeeze(1)  # (B,)

        # Double DQN
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)   # (B,36)
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)  # (B,1)

            next_q_target = self.target_net(next_states_t)
            next_q_target = next_q_target.gather(1, best_actions).squeeze(1)  # (B,)
            next_q_target[dones_t] = 0.0

        target = rewards_t + self.gamma * next_q_target

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
