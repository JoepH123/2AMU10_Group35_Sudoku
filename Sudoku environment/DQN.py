# dqn.py
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
    def __init__(self, input_shape=(9, 9), num_actions=81):
        super(CNNQNetwork, self).__init__()
        # Convolutionele lagen
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 9, 9)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 9, 9)
            nn.BatchNorm2d(64),  # Stabilisatie
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampled naar (64, 4, 4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Output: (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Fully connected lagen
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # Na MaxPooling op een 9x9 grid
            nn.ReLU(),
            nn.Linear(256, num_actions)  # Eindoutput, 81 acties
        )

    def forward(self, x):
        # Input: (batch_size, 1, 9, 9)
        x = self.conv(x)  # Convoluties
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected layers
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
        """
        Sla de originele transitie op en, als reward > 1, ook alle augmentaties.
        """
        # Originele transitie
        action_idx = action[0] * 9 + action[1]
        self.memory.push(state, action_idx, reward, next_state, done)

        # Voeg augmentatie toe alleen als reward > 1
        # if reward >= 0.1 or reward <= -0.1:
        #     augmented_transitions = generate_transformed_transitions(state, next_state, action)
        #     for (t_state, t_next_state, t_action) in augmented_transitions:
        #         t_action_idx = t_action[0] * 9 + t_action[1]
        #         self.memory.push(t_state, t_action_idx, reward, t_next_state, done)


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


