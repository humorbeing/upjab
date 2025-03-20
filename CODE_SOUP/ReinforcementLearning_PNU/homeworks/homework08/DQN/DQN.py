import gymnasium as gym
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

import copy
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):        
        
        self.action_size = action_size
        self.state_size = state_size

        # Hyper parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.quequeLenMax = 5000        
        self.train_start = 1000
        self.batch_size = 64
        self.train_epoch = 1

        self.memory = deque(maxlen=self.quequeLenMax)
        self.model = self.build_model()
        self.target_model = self.build_model()        
        self.update_target_model()               
        

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def build_model(self):
        model = Sequential()        
        model.add(Dense(64, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))        
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

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
        
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 살사의 큐함수 업데이트 식
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))
      
        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=self.train_epoch, verbose=0)


if __name__ == "__main__":
    EPISODE_LIMIT = 1000
    # env = gym.make("MountainCar-v0", render_mode="human")
    env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)    
    
    for episode in range(EPISODE_LIMIT):
        epo_step = 0
        done = False
        score = 0
        state, _info = env.reset()
        state = np.reshape(state, [1, agent.state_size])        

        while not done:                               
            action = agent.get_action(state)            
            next_state, reward, done, truncated, info = env.step(action)           
            next_state = np.reshape(next_state, [1, agent.state_size])            
            agent.append_sample(state, action, reward, next_state, done)
            score += reward
            
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = copy.deepcopy(next_state)            
            print(f'Episode {episode}, score {score}')
            if done:
                agent.update_target_model()
                

        






