import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reinforcement_learning_PNU.REINFORCE_environment import Env

EPISODE_LIMIT = 2

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class REINFORCEAgent:
    def __init__(self):        
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        # every step has -0.1 reward. Therefore no discount.
        self.discount_factor = 1.0  
        self.learning_rate = 0.01
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.states = []
        self.actions = []
        self.rewards = []        

    # 샘플 버퍼/메모리에 샘플을 추가
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # (입력값 = 상태), (출력값 = 큐함수)인 인공신경망(ANN) 생성
    def build_model(self):
        model = PolicyNetwork(self.state_size, self.action_size).to(self.device)
        print(model)
        return model

    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy = self.model(state_tensor).cpu().numpy().flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def train_model(self):
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        discounted_rewards_tensor = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # Get action probabilities
        action_probs = self.model(states_tensor)
        
        # Get log probabilities of selected actions
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())
        
        # Calculate loss (negative because we want to maximize)
        loss = -torch.mean(action_log_probs * discounted_rewards_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.states, self.actions, self.rewards = [], [], []

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        G_t = 0
        gamma = self.discount_factor

        for t in reversed(range(len(rewards))):
            G_t = G_t * gamma + rewards[t]
            discounted_rewards[t] = G_t
        
        return discounted_rewards


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = REINFORCEAgent()

    global_step = 0
    scores, episodes = [], []

    for episode in range(EPISODE_LIMIT):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        random_step = np.random.randint(0, 9)
        for _ in range(random_step):
            next_state, reward, done = env.step(4)
            state = np.reshape(next_state, [1, 15])

        while not done:            
            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)           

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            
            score += reward

            # state = next_state
            state = copy.deepcopy(next_state)            

            if done:
                # 에피소드마다 학습 결과 출력
                agent.train_model()                
                print("episode:", episode, "  score:", score, "global_step", global_step)

        
