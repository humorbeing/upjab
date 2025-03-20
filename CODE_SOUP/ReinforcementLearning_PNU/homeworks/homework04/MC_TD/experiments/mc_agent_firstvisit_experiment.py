import numpy as np
import random
from collections import defaultdict
from environment_experiment import Env

# 0 이면 every-visit
# 1 이면 first-visit
EveryVisit0_FirstVisit1 = 1


# 몬테카를로 에이전트 (모든 에피소드 각의 샘플로 부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions  # 모든 상태에서 동일한 set의 행동 선택 가능
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1  # epsilon-Greedy 정책
        self.samples = []  # 하나의 episode 동안의 기록을 저장하기 위한 버퍼/메모리
        self.value_table = defaultdict(float)  # 가치함수를 저장하기 위한 버퍼

    # 샘플 버퍼/메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # update_XXX 함수:
    # 모든 에피소드에서 에이전트가 "방문한 상태"의 가치함수를 업데이트
    # DP 기반의 방식에서는 모든 상태에 대해서 가치함수를 업데이트 했는데,
    # Monte Carlo 에서는 직접 방문한 상태에 대해서만 가치함수를 업데이트 함
    def update_EveryVisit(self):
        """
            1. 여기를 구현하세요
        """
        length = len(self.samples)
        G_list = []
        for i in range(length):
            reverse_index = length-1-i
            record = self.samples[reverse_index]            
            if record[2]:
                value = record[1]
                G_list.append((record[0], value))
            else:
                value = record[1] + self.discount_factor * value
                G_list.append((record[0], value))
        
        for i in range(length):
            reverse_index = length-1-i
            state, G = G_list[reverse_index]
            V = self.value_table[str(state)]
            alpha = self.learning_rate
            V_new = V + alpha * (G - V)
            self.value_table[str(state)] = V_new

    def update_FirstVisit(self):
        """
            2. 여기를 구현하세요
        """
        length = len(self.samples)
        G_list = []
        for i in range(length):
            reverse_index = length-1-i
            record = self.samples[reverse_index]            
            if record[2]:
                value = record[1]
                G_list.append((record[0], value))
            else:
                value = record[1] + self.discount_factor * value
                G_list.append((record[0], value))
        
        check = set()
        for i in range(length):
            reverse_index = length-1-i
            state_, G = G_list[reverse_index]
            state = str(state_)
            if state in check:
                pass
            else:
                check.add(state)
                V = self.value_table[state]
                alpha = self.learning_rate
                V_new = V + alpha * (G - V)
                self.value_table[state] = V_new
            

    # 상태-가치함수에 따라서 행동을 결정
    # 다음 time-step 때 선택할 수 있는 상태들 중에서, 가장 큰 가치함수 값을 리턴하는 상태로 이동
    # 입실론 탐욕 정책을 사용
    def get_action(self, state_):
        """
            3. 여기를 구현하세요
        """
        roll = random.random()        
        if roll < self.epsilon:
            # explore
            action = random.choice(self.actions)
        else:
            # exploit
            next_state = self.possible_next_state(state_)
            action = self.arg_max(next_state)
        
        return action

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    # => 정책 (pi)은 없지만, 최적의 정책을 유도하는 역할을 하는 함수
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # 현재 상태가 state 일때, 다음 상태가 될 수 있는 모든 상태에 대한 가치함수 계산
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]

        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]

        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]

        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state


def get_return(rewards):
    l = len(rewards)
    ret = 0
    gamma = 0.9
    for i in range(l):
        reverse_index = l-1-i
        ret = rewards[reverse_index] + gamma * ret
    return ret
import logging
logging.basicConfig(filename='logs/firstvisit.log', level=logging.INFO, filemode='w')  # filemode='w'
logging.info("starting")
from tqdm import tqdm
# 메인 함수
if __name__ == "__main__":
    ROUND = 999
    for r in tqdm(range(ROUND)):
        env = Env()
        agent = MCAgent(actions=list(range(env.n_actions)))
        env.is_render = False
        MAX_EPISODES = 1000  # 최대 에피소드 수
        for episode in range(MAX_EPISODES):
            state = env.reset()  # 에피소드 시작 : 환경을 초기화하고, 상태 = 초기상태로 설정
            action = agent.get_action(state)
            rewards = []
            
            while True:
                env.render()  # 화면 그리기

                # action 행동을 하고 다음 상태로 이동
                # 보상은 숫자이고, 완료 여부는 boolean
                next_state, reward, done = env.step(action)
                rewards.append(reward)
                # 획득한 샘플을 샘플 버퍼/메모리에 저장
                # 에피소드가 끝나야 리턴값을 알 수 있으므로, done=True 일때까지 버퍼에 보관
                agent.save_sample(next_state, reward, done)

                # env.step(action)을 통해 상태가 변경 되었고,
                # 변경된 상태에서 택할 행동을 결정
                action = agent.get_action(next_state)

                # 에피소드가 완료되었다면, 가치함수 업데이트
                if done:
                    
                    if EveryVisit0_FirstVisit1 is 0:
                        agent.update_EveryVisit()
                    else:
                        agent.update_FirstVisit()
                    agent.samples.clear()
                    ret = get_return(rewards)
                    log_s = f'round |{r}| episode |{episode}| return |{ret}|'
                    logging.info(log_s)
                    break
        env.destroy()

