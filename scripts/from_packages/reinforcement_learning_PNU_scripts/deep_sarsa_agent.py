import copy

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reinforcement_learning_PNU.REINFORCE_environment import Env


EPISODE_LIMIT = 2
PRINT_STATE = True
PRINT_STATE = False
#PRINT_ACTION = True
PRINT_ACTION = False


# Q-Network 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 확장된 그리드월드에서의 딥살사 에이전트
class DeepSarsaAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        
        # PyTorch 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # 저장된 학습 결과를 불러오기
        if self.load_model:
            self.epsilon = 0.05
            self.model.load_state_dict(torch.load('save_model/deep_sarsa_trained.pt'))
            self.model.eval()


    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()


    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.model.train()
        
        # numpy array를 PyTorch 텐서로 변환
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        
        # 현재 Q값 예측
        current_q_values = self.model(state_tensor)
        
        # 타겟 Q값 계산
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
        
        # 살사의 큐함수 업데이트 식
        target = current_q_values.clone()
        if done:
            target[0][action] = reward
        else:
            target[0][action] = (reward + self.discount_factor * 
                                next_q_values[0][next_action])
        
        # 손실 계산 및 역전파
        loss = self.criterion(current_q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSarsaAgent()

    global_step = 0
    scores, episodes = [], []

    for episode in range(EPISODE_LIMIT):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])
        if PRINT_STATE:
            print('[init] state : ' + str(state))

        while not done:
            env.render()
            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            if PRINT_ACTION:
                print('action : ' + str(action))

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            score += reward

            # 획득한 <s,a,r,s',a'> 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action, done)

            # state = next_state
            state = copy.deepcopy(next_state)

            if PRINT_STATE:
                print('state : ' + str(state))

            if done:
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(episode)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("save_graph/deep_sarsa_.png")
                print("episode:", episode, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        # 100 에피소드마다 모델 저장
        # if episode % 100 == 0:
        #     torch.save(agent.model.state_dict(), "save_model/deep_sarsa.pt")
