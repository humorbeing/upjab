import copy
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

# fix error
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from env import Env


EPISODE_LIMIT = 500


class REINFORCEAgent:
    def __init__(self):
        
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        
        self.discount_factor = 1.0
        self.learning_rate = 0.005   

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        
        
        self.states = []
        self.actions = []
        self.rewards = []

    # 샘플 버퍼/메모리에 샘플을 추가
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # (입력값 = 상태), (출력값 = 큐함수)인 인공신경망(ANN) 생성
    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(64, input_dim=self.state_size, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     model.add(Dense(32, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     # output layer
    #     model.add(Dense(self.action_size, activation='softmax',
    #                     kernel_initializer='he_uniform'))
    #     model.summary()
        
    #     return model

    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(128, input_dim=self.state_size, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     model.add(Dense(64, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     model.add(Dense(32, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     # output layer
    #     model.add(Dense(self.action_size, activation='softmax',
    #                     kernel_initializer='he_uniform'))
    #     model.summary()
        
    #     return model
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu',
                        kernel_initializer='he_uniform'))
        # output layer
        model.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        model.summary()
        
        return model


    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    

    def build_optimizer(self):
        action_onehot = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None,])

        action_prob = self.model.output
        action_log_prob = K.log(K.sum(action_prob * action_onehot, axis=1))
        to_maximize = K.mean(action_log_prob * discounted_rewards)
        loss = to_maximize * (-1)

        optimizer = Adam(lr=self.learning_rate)
        
        # fix error
        # import tensorflow as tf
        # tf.compat.v1.disable_eager_execution()
        updates = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss)
        
        train = K.function(
            inputs=[self.model.input,
            action_onehot,
            discounted_rewards],
            outputs=[self.model.output],
            updates=updates)
        return train


    def train_model(self):
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        states = np.vstack(self.states)
        actions = np.zeros([len(self.actions), self.action_size])
        for i, action in enumerate(actions):
            action[self.actions[i]] = 1       

        self.optimizer([states, actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        G_t = 0
        gamma = self.discount_factor

        for t in reversed(range(len(rewards))):
            G_t = G_t * gamma + rewards[t]
            discounted_rewards[t] = G_t
        
        return discounted_rewards


import logging
logging.basicConfig(filename='logs/m2d1.log', level=logging.INFO, filemode='w')  # filemode='w'



env = Env()
# from tqdm import tqdm
MAX_restart = 100000
r = 1
ROUND = 20
broken_count = 0
while True:
    agent = REINFORCEAgent()
    global_step = 0    
    log_list = []
    is_broken = False

    for episode in range(EPISODE_LIMIT):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        # random_step = np.random.randint(0, 9)
        # for _ in range(random_step):
        #     next_state, reward, done = env.step(4)
        #     state = np.reshape(next_state, [1, 15])

        while not done:            
            global_step += 1
            print(f'round: {r}. global_step: {global_step}. broken count: {broken_count}')
            if global_step > MAX_restart:
                is_broken = True
                break
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
                episodic_length = len(agent.rewards)
                agent.train_model()        
               
                log_ = f'round |{r}| episode |{episode}| on_return |{score}| on_length |{episodic_length}| global_step |{global_step}|'
                # logging.info(log_)
                log_list.append(log_)
                print(log_)


        if is_broken:
            break
    
    if is_broken:
        broken_count = broken_count + 1
        pass
    else:
        for one_log in log_list:
            logging.info(one_log)
        
        r = r + 1
    
    if r == ROUND:
        break

    if broken_count == 100:
        break

        
logging.info(f'broken count: {broken_count}')


        