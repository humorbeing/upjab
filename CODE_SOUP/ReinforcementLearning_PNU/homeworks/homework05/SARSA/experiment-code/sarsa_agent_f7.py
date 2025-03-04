import numpy as np
import random
from collections import defaultdict
from env_exp import Env

MAX_EPISODE = 1000

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 3) 시간이 지날수록 e 값이 감소하도록 코드를 수정하세요.
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.current_epsilon = None

    # 큐함수 업데이트
    def learn(self, sample):
        """
            2) 여기에 들어갈 내용을 구현하세요
        """
        alpha = self.learning_rate
        gamma = self.discount_factor
        current_state, current_action, R, \
            next_state, next_action, done = sample
        Q = self.q_table[str(current_state)][current_action]
        Q_next = self.q_table[str(next_state)][next_action]
        
        Q_new = Q + alpha * (R + gamma * Q_next - Q)
        self.q_table[str(current_state)][current_action] = Q_new
        if done:      
            Q_end = Q_next + alpha * (R - Q_next)
            for _action in self.actions:
                self.q_table[str(next_state)][_action] = Q_end
            
        # print('end')

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state, episode):
        x = (MAX_EPISODE - episode) / 1001
        epsilon_max = 0.3
        epsilon_min = 0.01
        y = x
        # y = x**3
        # y = 1/(1+ (x/(1-x))**(-3))
        epsilon = y * (epsilon_max - epsilon_min) + epsilon_min
        # epsilon = self.epsilon
        self.current_epsilon = epsilon
        if np.random.rand() < epsilon:
            # 무작위 행동 선택 (exploration)
            best_action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 최적 행동 반환 (exploitation)
            state_action = self.q_table[state]
            best_action = self.arg_max(state_action)
        return best_action
        # state_action = np.array(self.q_table[state])
        # exp_action = np.exp(state_action / T)
        # prob_action = exp_action / exp_action.sum()
        # best_action = np.random.choice(self.actions,p=prob_action)
        # return best_action
    
    def evaluate_action(self, state, episode):        
        epsilon = self.epsilon        
        if np.random.rand() < epsilon:
            # 무작위 행동 선택 (exploration)
            best_action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 최적 행동 반환 (exploitation)
            state_action = self.q_table[state]
            best_action = self.arg_max(state_action)
        return best_action
        

    """
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
    """

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = -9999
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

def get_return(rewards):
    l = len(rewards)
    ret = 0
    gamma = 0.9
    for i in range(l):
        reverse_index = l-1-i
        ret = rewards[reverse_index] + gamma * ret
    return ret


def eval(env, agent, limit=100):
    state = env.reset()
    env.is_render = False
    rewards = []
    # 현재 상태에서 어떤 행동을 할지 선택
    action = agent.evaluate_action(str(state), episode)    
    # 한개의 episode를 처음부터 끝까지 처리하는 while-loop

    while True:
        next_state, next_reward, done = env.step(action)
        next_action = agent.evaluate_action(str(next_state), episode)        
        rewards.append(next_reward)
        length_episode = len(rewards)        
        
        state = next_state
        action = next_action
        if done:
            ret = get_return(rewards)                        
            break

        if length_episode == limit:
            ret = -999
            break
    
    return ret, length_episode

import logging
logging.basicConfig(filename='logs/f7.log', level=logging.INFO, filemode='w')  # filemode='w'
from tqdm import tqdm
if __name__ == "__main__":
    ROUND = 999
    env = Env()
    env.is_render = False
    for r in tqdm(range(ROUND)):
        try:
            # 환경에 대한 instance 생성
            agent = SARSAgent(actions=list(range(env.n_actions)))  # Sarsa Agent 객체 생성
            
            # 지정된 횟수(MAX_EPISODE)만큼 episode 진행
            for episode in range(MAX_EPISODE):
                # 게임 환경과 상태를 초기화 하고, 상태(state)값 얻기
                state = env.reset()
                rewards = []
                # 현재 상태에서 어떤 행동을 할지 선택
                action = agent.get_action(str(state), episode)
                
                # 한개의 episode를 처음부터 끝까지 처리하는 while-loop
                while True:
                    env.render()

                    """
                        1) 여기에 들어갈 내용을 구현하세요.
                    """
                    
                    next_state, next_reward, done = env.step(action)
                    next_action = agent.get_action(str(next_state), episode)
                    sample = [state, action, next_reward,
                            next_state, next_action, done]
                    rewards.append(next_reward)
                    
                    agent.learn(sample)
                    
                    state = next_state
                    action = next_action


                    # 모든 큐함수 값을 화면에 표시
                    env.print_value_all(agent.q_table)

                    # episode가 끝났으면 while-loop을 종료
                    if done:
                        on_return = get_return(rewards)
                        on_length = len(rewards)
                        evl_return, evl_length = eval(env, agent)
                        log_s = f'round |{r}| episode |{episode}| on_return |{on_return}| on_length |{on_length}| epsilon |{agent.current_epsilon}| Eval_return |{evl_return}| Eval_length |{evl_length}|'
                        # print(log_s)
                        logging.info(log_s)
                        break
        except:
            print('-----------something broken. continue------------')
        

