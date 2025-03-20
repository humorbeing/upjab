import numpy as np
import random
from env_exp import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01  # 학습률, 2)번 문제
        self.discount_factor = 0.9  # 감가율, 3)번 문제
        self.epsilon = 0.05  # 랜덤 행동을 할 확률
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        """
            1) 여기에 들어갈 코드를 작성하세요.
        """
        alpha = self.learning_rate
        gamma = self.discount_factor
        Q = self.q_table[state][action]
        state_action = self.q_table[next_state]
        next_action = self.arg_max(state_action)
        Q_next = self.q_table[next_state][next_action]
        Q_new = Q + alpha*(reward + gamma*Q_next - Q)
        self.q_table[state][action] = Q_new        

    # 입실론 탐욕 정책에 따라서 행동을 선택
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        return action

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

def get_return(rewards):
    l = len(rewards)
    ret = 0
    gamma = 0.9
    for i in range(l):
        reverse_index = l-1-i
        ret = rewards[reverse_index] + gamma * ret
    return ret

import logging
logging.basicConfig(filename='logs/gamma.log', level=logging.INFO, filemode='w')  # filemode='w'
from tqdm import tqdm
import time
if __name__ == "__main__":    
    gamma_list1 = np.linspace(0.01, 1.0, 20)
    gamma_list2 = np.linspace(1.0, 3.0, 4)
    gamma_list = np.concatenate((gamma_list1, gamma_list2[1:]))    
    env = Env()
    env.is_render = False
    for ga in gamma_list:
        print(f'gamma: {ga}')
        ROUND = 999
        
        start_time = time.time()
        for r in tqdm(range(ROUND)):
            try:
                agent = QLearningAgent(actions=list(range(env.n_actions)))  # Q러닝 Agent 객체 생성            
                agent.discount_factor = ga
                EPISODE_MAX = 1000
                for episode in range(EPISODE_MAX):
                    state = env.reset()  # 환경을 초기화 하고, 초기 상태 s 를 얻기.
                    rewards = []
                    while True:  # 현재 episode가 끝날 때 까지 반복
                        env.render()

                        # 현재 상태에 대한 행동 선택
                        action = agent.get_action(str(state))
                        # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
                        next_state, reward, done = env.step(action)
                        rewards.append(reward)
                        # <s,a,r,s'> 샘플로 Q 함수를 업데이트
                        agent.learn(str(state), action, reward, str(next_state))
                        state = next_state

                        # 모든 큐함수를 화면에 표시
                        env.print_value_all(agent.q_table)

                        if done:  # 현재 에피소드가 끝난경우...
                            episodic_length = len(rewards)
                            episodic_return = get_return(rewards)
                            log_s = f'round |{r}| episode |{episode}| on_return |{episodic_return}| on_length |{episodic_length}| alpha |{agent.learning_rate}| gamma |{agent.discount_factor}|'
                            logging.info(log_s)
                            break  # while-loop를 탈출
            
            except:
                print('broken, keep going')
        
        end_time = time.time()

        duration = end_time - start_time
        logging.info(f'duration |{duration}| alpha |{agent.learning_rate}| gamma |{agent.discount_factor}|')
        print('end')

