import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

def generate_transformed_transitions(state, next_state, action):
    transformations = []
    (row, col) = action

    for k in range(4):
        # Pas rotatie toe op elk kanaal afzonderlijk
        rotated_state = np.array([np.rot90(channel, k) for channel in state])
        rotated_next_state = np.array([np.rot90(channel, k) for channel in next_state])

        if rotated_state.shape != state.shape or rotated_next_state.shape != next_state.shape:
            raise ValueError(f"Shape mismatch after rotation {k}: {rotated_state.shape}")

        if k == 0:
            r_a, c_a = row, col
        elif k == 1:
            r_a, c_a = col, 8 - row
        elif k == 2:
            r_a, c_a = 8 - row, 8 - col
        elif k == 3:
            r_a, c_a = 8 - col, row

        transformations.append((rotated_state, rotated_next_state, (r_a, c_a)))

        # Pas flip toe op elk kanaal afzonderlijk
        flipped_state = np.array([np.flip(channel, axis=1) for channel in rotated_state])
        flipped_next_state = np.array([np.flip(channel, axis=1) for channel in rotated_next_state])

        if flipped_state.shape != state.shape or flipped_next_state.shape != next_state.shape:
            raise ValueError(f"Shape mismatch after flip: {flipped_state.shape}")

        transformations.append((flipped_state, flipped_next_state, (r_a, 8 - c_a)))

    return transformations


class CNNQNetwork(nn.Module):
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
        # Flatten: 64*9*9 = 5184
        self.fc = nn.Sequential(
            nn.Linear(5184, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
    def __init__(self, lr=5e-4, gamma=0.99, batch_size=128, replay_size=100000, update_target_every=500, tau=0.005):
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
        # action_idx = action[0] * 9 + action[1]
        # self.memory.push(state, action_idx, reward, next_state, done)

        #Als je augmentaties wilt gebruiken, activeer dit blok:
        augmented_transitions = generate_transformed_transitions(state, next_state, action)
        for (t_state, t_next_state, t_action) in augmented_transitions:
            t_action_idx = t_action[0] * 9 + t_action[1]
            self.memory.push(t_state, t_action_idx, reward, t_next_state, done)

    def soft_update(self):
        # Soft update: target_param ← tau * policy_param + (1 - tau) * target_param
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )


    def update(self):
        # Zorg dat er voldoende samples in het replay-buffer zitten
        if len(self.memory) < self.batch_size:
            return

        # Haal een batch op uit het replay-buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Converteer de numpy-arrays naar Torch tensors
        states_np = np.stack(states)
        next_states_np = np.stack(next_states)

        states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Schakel naar trainingsmodus en bereken de Q-waarden voor de huidige states
        self.policy_net.train()
        q_values = self.policy_net(states_t)       # Vorm: (B, 81) 
        q_values = q_values.gather(1, actions_t).squeeze(1)  # Vorm: (B,)

        # Bereken de target Q-waarden met Double DQN-logica
        with torch.no_grad():
            # Kies de beste acties volgens de policy_net
            next_q_policy = self.policy_net(next_states_t)       # Vorm: (B, 81)
            next_actions = next_q_policy.argmax(dim=1).unsqueeze(1)  # Vorm: (B,1)

            # Haal de Q-waarden van deze acties uit de target_net
            next_q_target = self.target_net(next_states_t)       # Vorm: (B, 81)
            next_q_target = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)
            next_q_target[dones_t] = 0.0  # Done-states krijgen Q=0

        # Standaard DQN target
        target = rewards_t + self.gamma * next_q_target

        # MSE-loss
        loss = F.mse_loss(q_values, target)

        # Optimaliseer de policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Houd aantal updates bij
        self.steps_done += 1

        # Voer een zachte update uit (soft target update)
        self.soft_update()
