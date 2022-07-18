import gym
import gym_stag_hunt
# Possible Moves
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

random_seed = 42
# if setting seed the result is always the same
# np.random.seed(random_seed)
# random.seed(random_seed)
# tf.random.set_seed(random_seed)

def valid_action(position, action, grid_size):
  if action == LEFT:
    return position[1] > 0
  elif action == UP:
    return position[0] > 0
  elif action == RIGHT:
    return position[1] < (grid_size[1]-1)
  elif action == DOWN:
    return position[0] < (grid_size[0]-1)
  return False

def getValidActions(position, grid_size):
  actions = [UP, LEFT, DOWN, RIGHT]
  valid_actions = []
  for a in actions:
    if valid_action(position, a, grid_size):
      valid_actions.append(a)
        
  return valid_actions

def randomAction(position, grid_size=(5,5)):
  valid_actions = getValidActions(position, grid_size)
  randomAction = random.choices(valid_actions)[0]
  return randomAction

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def vec2mat(coords_state, grid_size=(5,5)):
  r = np.zeros(grid_size)
  g = np.zeros(grid_size)
  b = np.zeros(grid_size)
  # computer pos    
  b[coords_state[1], coords_state[0]] += 1
  # human pos
  r[coords_state[3], coords_state[2]] += 1
  # stag pos
  r[coords_state[5], coords_state[4]] += 0.8039
  g[coords_state[5], coords_state[4]] += 0.498
  b[coords_state[5], coords_state[4]] += 0.1961
  # plants pos
  for i in range(6, 12, 2):
      g[coords_state[i+1], coords_state[i]] += 1

  # plt.imshow(np.dstack((r,g,b)))
  # plt.show()
  return NormalizeData(np.dstack((r,g,b)))

def combine_following_states(prev, current):
  r2, g2, b2 = current[:, :, 0], current[:, :, 1], current[:, :, 2]
  human_pos = np.where(r2 == 1)
  computer_pos = np.where(b2 == 1)
  bushes_pos = np.where(g2 == 1)
  stag_pos = np.where(r2 == 0.8039, g2 == 0.498, b2 == 0.1961)
  
  new_cell = prev * 0.75

  new_cell[:, :, 0][human_pos] = 1
  new_cell[:, :, 1][bushes_pos] = 1
  new_cell[:, :, 2][computer_pos] = 1

  new_cell[:, :, 0][stag_pos] += 0.8039
  new_cell[:, :, 1][stag_pos] += 0.498
  new_cell[:, :, 2][stag_pos] += 0.1961

  return new_cell


class Follow_Stag:
  def __init__(self, grid_size=(5,5)):
    self.grid_size = grid_size
    self.stag_pos = [2,2]
    self.human_pos = [0,0]
  
  def update_pos(self, coords_state):
    self.human_pos = [coords_state[3], coords_state[2]]
    self.stag_pos = [coords_state[5], coords_state[4]]

  def takeActionTo(self, current, to):
    if current[0] < to[0]:
        return DOWN #down
    elif current[0] > to[0]:
        return UP #up
    elif current[1] < to[1]:
        return RIGHT #right
    elif current[1] > to[1]:
        return LEFT #left
    return randomAction(self.human_pos) #takes random action
  
  def predict_action(self):
    return self.takeActionTo(self.human_pos, self.stag_pos)
  

class HumamModel:
  def __init__(self):
    self.model = tf.keras.models.load_model('./data/humanModel/empathy_model_v0.h5')

  def predict_action(self, state):
    position = np.where(state[:, :, 0] == np.amax(state[:, :, 0]))
    state = tf.expand_dims(state, 0)  # Create a batch
    score = self.model.predict(state)[0][0]

    #fix numeric problem that softmax not always sum to 1
    diff = 1 - sum(score)
    score = score + diff/len(score)
    dict_scores = dict(enumerate(score))
    action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    while(not valid_action(position, action, (5,5))):
        del dict_scores[action]
        action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    return action

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, initial_epsilon):
        
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.95 # discount rate
        self.epsilon = initial_epsilon # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.batch_size = 128
        self.train_start = 1000 # memory_size

        # defining model parameters
        self.ddqn = True
        self.Soft_Update = True
        self.distribution = True

        self.TAU = 0.1 # target network soft update hyperparameter
        
        self.state_size = state_size
        self.action_size = action_size

        # create main model
        self.model = self.OurModel(input_shape=self.state_size, action_space = action_size)
        self.target_model = self.OurModel(input_shape=self.state_size, action_space = action_size)

    def OurModel(self, input_shape, action_space):
      X_input = Input(shape=input_shape)
      X = X_input
      X = Conv2D(filters=4, kernel_size=(4,4), padding='same', activation='relu')(X)
      X = Conv2D(filters=8, kernel_size=(4,4), padding='same', activation='relu')(X)
      X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
      X = MaxPool2D()(X)
      X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
      X = Flatten()(X)
      # X = Dense(256, activation='relu')(X)
      X = Dense(32, activation='relu')(X)
      # Output Layer with # of actions: 5 nodes (left, right, up, down, stay)
      X = Dense(action_space, activation="linear")(X)

      model = Model(inputs = X_input, outputs = X)
      model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.00001), metrics=["accuracy"])
      return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
      if np.random.random() <= self.epsilon:
        position = np.where(state[:, :, 2] == np.amax(state[:, :, 2]))
        return randomAction(position)
      else:
        state = tf.expand_dims(state, 0)  # Create a batch
        return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))
        
        state = np.zeros((self.batch_size,) + self.state_size)
        next_state = np.zeros((self.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target,  epochs=2, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

class Monitor:
  def __init__(self):
    # Plot agent performance
    self.scores, self.human_scores, self.episodes, self.averages = [], [], [], []
    fig, self.ax1 = plt.subplots(1, 1, figsize=(18, 9))
    self.ax1.set_ylabel('Score', fontsize=15)
    self.ax1.set_xlabel('Episode', fontsize=15)
    
  def PlotModel(self, score, human_score, episode, agent_name):
    window_size = 50
    self.scores.append(score)
    self.human_scores.append(human_score)
    self.episodes.append(episode)        
    if len(self.scores) > window_size:
        # moving avrage:
        self.averages.append(sum(self.scores[-1 * window_size: ]) / window_size)
    else:
        self.averages.append(sum(self.scores) / len(self.scores))

    self.ax1.plot(self.scores, 'b')
    # self.ax1.plot(self.human_scores, 'r')
    self.ax1.plot(self.averages, 'r')

    try:
        plt.savefig("data/images/"+agent_name+".png", dpi = 150)
    except OSError:
        pass

    return str(self.averages[-1])[:5]

def run(env, episodes = 1100, epsilon_decay=0.9975, train=False, beta = 0.5, SARL=False):
  human_model = HumamModel()
  # human_model = Follow_Stag()
  if not train:
    agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 0)
    if SARL:
      agent_model.load("data/weights/empathy_SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+".h5")
    else:
      agent_model.load("data/weights/empathy_ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+".h5")
    episodes = 5
  else:
    agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 1.0)
  monitor = Monitor()

  for ep in range(episodes):    
    obs = env.reset()
    # human_model.update_pos(obs)
    obs = vec2mat(obs)
    if not train:
      env.render()
    
    episodes_per_game = 60
    ep_reward = 0
    ep_human_reward = 0
    ep_SARL_reward = 0
    for _ in range(episodes_per_game):
      computer_action = agent_model.act(obs)
      human_action = human_model.predict_action(obs)
      next_obs, rewards, done, info = env.step([computer_action, human_action])
      next_obs = next_obs[0]
      # human_model.update_pos(next_obs)
      next_obs = vec2mat(next_obs)
      # next_obs = combine_following_states(obs, next_obs)
      agent_reward = rewards[0]
      SARL_reward = beta * agent_reward + (1 - beta) * rewards[1]
      if train:
        if SARL:
          agent_model.remember(np.expand_dims(obs, axis=0), computer_action, SARL_reward, np.expand_dims(next_obs, axis=0), done)
        else:
          agent_model.remember(np.expand_dims(obs, axis=0), computer_action, agent_reward, np.expand_dims(next_obs, axis=0), done)
        agent_model.replay()
      else:
        time.sleep(0.2)
        env.render()
      obs = next_obs

      ep_reward += agent_reward
      ep_human_reward += rewards[1]
      ep_SARL_reward += SARL_reward
      
    if train:
      # every step update target model
      agent_model.update_target_model()
      # decay epsilon
      agent_model.updateEpsilon()
      # every episode, plot the result
      if SARL:
        average = monitor.PlotModel(ep_reward, ep_human_reward, ep, "empathy_SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay))
      else:
        average = monitor.PlotModel(ep_reward, ep_human_reward, ep, "empathy_ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay))
      print("episode: {}/{}, score: {:.2}, average: {}, e: {:.3}, SARL score: {}".format(ep, episodes, ep_reward, average, agent_model.epsilon, ep_SARL_reward))
    else:
      print("episode: {}/{}, score: {:.2}, SARL score: {}".format(ep, episodes, ep_reward, ep_SARL_reward))
    ep_reward = 0
    ep_human_reward = 0
    ep_SARL_reward = 0

  if train:
    if SARL:
      agent_model.save("data/weights/empathy_SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+".h5")
    else:
      agent_model.save("data/weights/empathy_ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+".h5")
  env.close()

if __name__ == "__main__":
  env = gym.make("StagHunt-Hunt-v0", obs_type='coords', load_renderer= True, enable_multiagent=True, forage_quantity=3) # you can pass config parameters here

  #train dqn agent
  run(env, episodes=4000, epsilon_decay = 0.9995, train=True, SARL=False)
  #test dqn agent
  # run(env, episodes=4000, epsilon_decay = 0.9995, train=False, SARL=False)