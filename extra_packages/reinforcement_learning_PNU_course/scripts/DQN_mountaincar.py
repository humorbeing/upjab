# pip install "gymnasium[classic_control]"
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import numpy as np
import random
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

log_name = 'v08'
MAX_EPISODE = 1000

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, action_size)
        
        # He initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):        
        
        self.action_size = action_size
        self.state_size = state_size

        # Hyper parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        # self.epsilon = 1.  # exploration
        # self.epsilon_decay = 0.9999
        # self.epsilon_min = 0.01
        self.epsilon = 0.05  # exploration        
        self.epsilon_decay = 1.
        self.epsilon_min = 0.01
        self.x_weight = 1.2
        self.quequeLenMax = 2000    
        self.train_start = 2000
        self.batch_size = 2000
        self.train_epoch = 1
        self.clear_memory = 0

        self.memory = deque(maxlen=self.quequeLenMax)
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        print(self.model)
        

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()


    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)
        
        # Train for specified epochs
        self.model.train()
        for epoch in range(self.train_epoch):
            # Get current Q values
            current_q_values = self.model(states_tensor)
            
            # Get target Q values
            with torch.no_grad():
                next_q_values = self.target_model(next_states_tensor)
                max_next_q_values = next_q_values.max(1)[0]
            
            # Compute target Q values
            target_q_values = current_q_values.clone()
            for i in range(self.batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards_tensor[i]
                else:
                    target_q_values[i][actions[i]] = rewards_tensor[i] + self.discount_factor * max_next_q_values[i]
            
            # Compute loss and update
            loss = self.criterion(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        for _ in range(self.clear_memory):
            self.memory.popleft()
        


if __name__ == "__main__":
    # import logging
    # logging.basicConfig(filename=f'logs/{log_name}.log', level=logging.INFO, filemode='w')
    
    env = gym.make("MountainCar-v0", render_mode="human")
    # env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    ROUND = 1
    update_model = 2000
    count_update = 0
    for _round in range(ROUND):

        agent = DQNAgent(state_size, action_size)    
        
        for episode in range(MAX_EPISODE):
            epo_step = 0
            done = False
            score = 0
            state, _info = env.reset()
            state = np.reshape(state, [1, agent.state_size])        

            while not done:                               
                action = agent.get_action(state)            
                next_state, reward, done, truncated, info = env.step(action)           
                next_state = np.reshape(next_state, [1, agent.state_size])
                x_pos = next_state[0][0]
                v_ = next_state[0][1]/0.07
                zero_x = x_pos + 0.5
                make_reward = 2 * zero_x * v_
                if done:
                    make_reward = make_reward + 2
                agent.append_sample(state, action, make_reward, next_state, done)
                score += reward
                
                if len(agent.memory) >= agent.train_start:
                    agent.epsilon = 0.05
                    agent.train_model()
                else:
                    agent.epsilon = 1
                count_update = count_update + 1
                if count_update == update_model:
                    agent.update_target_model()
                    count_update = 0

                state = copy.deepcopy(next_state)            
                print(f'round |{_round}| episode |{episode}| score |{score}| epsilon {agent.epsilon:0.04f} x {zero_x:0.04f} v {v_:0.04f} mr {make_reward}')
                log_s = f'round |{_round}| episode |{episode}| score |{score}| pos |{zero_x}| v |{v_}| RW |{make_reward}|'
                # logging.info(log_s)
                if done:
                    
                    agent.update_target_model()
                

        






