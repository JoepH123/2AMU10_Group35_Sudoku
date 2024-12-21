import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

def generate_transformed_transitions(state, next_state, action):
    """
    Genereer alle symmetrische transformaties (4 rotaties en horizontale spiegelingen)
    voor state, next_state en pas ook de actie aan.
    Retourneert een lijst van (transformed_state, transformed_next_state, transformed_action).
    """
    transformations = []
    (row, col) = action

    for k in range(4):
        # Rotatie k * 90 graden
        rotated_state = np.rot90(state, k)
        rotated_next_state = np.rot90(next_state, k)

        # Actie aanpassen voor rotatie
        if k == 0:
            r_a, c_a = row, col
        elif k == 1:
            r_a, c_a = col, 8 - row
        elif k == 2:
            r_a, c_a = 8 - row, 8 - col
        elif k == 3:
            r_a, c_a = 8 - col, row

        # Toevoegen originele (na rotatie)
        transformations.append((rotated_state, rotated_next_state, (r_a, c_a)))

        # Horizontale spiegeling van zowel rotated_state als rotated_next_state
        flipped_state = np.flip(rotated_state, axis=1)
        flipped_next_state = np.flip(rotated_next_state, axis=1)

        # Actie aanpassen voor flip (flip horizontaal: col -> 8 - c_a)
        transformations.append((flipped_state, flipped_next_state, (r_a, 8 - c_a)))

    return transformations

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape=(9,9), num_actions=81):
        super(CNNQNetwork, self).__init__()
        # input: (3,9,9)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Na 3 conv-lagen (64,9,9) => Flatten 64*9*9 = 5184
        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        # x shape: (B,3,9,9)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),
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
        # state shape: (3,9,9)
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3,9,9)
            q_values = self.policy_net(state_t).cpu().numpy().squeeze() # (81,)

        mask = np.ones(81) * (-1e9)
        for (r,c) in valid_actions:
            action_idx = r*9 + c
            mask[action_idx] = q_values[action_idx]

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            best_action_idx = np.argmax(mask)
            action = (best_action_idx // 9, best_action_idx % 9)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        action_idx = action[0] * 9 + action[1]
        self.memory.push(state, action_idx, reward, next_state, done)

        # Indien je augmentaties wilt gebruiken, kun je dit stuk weer activeren:
        # if abs(reward) > 0.1:
        #     augmented_transitions = generate_transformed_transitions(state, next_state, action)
        #     for (t_state, t_next_state, t_action) in augmented_transitions:
        #         t_action_idx = t_action[0] * 9 + t_action[1]
        #         self.memory.push(t_state, t_action_idx, reward, t_next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Converteer naar numpy arrays
        states_np = np.stack(states)        # (B,3,9,9)
        next_states_np = np.stack(next_states) # (B,3,9,9)

        states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)  # (B,3,9,9)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self.device) # (B,3,9,9)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1) # (B,1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Current Q values
        self.policy_net.train()
        q_values = self.policy_net(states_t) # (B,81)
        q_values = q_values.gather(1, actions_t).squeeze(1) # Q(s,a) => (B,)

        # Double Q-learning:
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states_t) # (B,81)
            next_actions = next_q_values_policy.argmax(dim=1).unsqueeze(1) # (B,1)
            next_q_values_target = self.target_net(next_states_t)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)
            next_q_values[dones_t] = 0.0

        target = rewards_t + self.gamma * next_q_values
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


