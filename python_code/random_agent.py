import gym
import gym_stag_hunt

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import time

import tensorflow as tf

# Possible Moves
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND


env = gym.make("StagHunt-Hunt-v0", obs_type='coords', load_renderer= True, enable_multiagent=True, forage_quantity=3) # you can pass config parameters here
# replace the computer initial position with the human position

class HumamModel:
  def __init__(self, grid_size=(5,5)):
    self.model = tf.keras.models.load_model('./data/humanModel/model_v0.h5')
    self.grid_size = grid_size

  def vec2mat(self, row):
      r = np.ones(self.grid_size)
      g = np.ones(self.grid_size)
      b = np.ones(self.grid_size)

      # computer pos
      pos = (row[1], row[0])
      r[pos] = 0
      g[pos] = 0

      # human pos
      pos = (row[3], row[2])
      b[pos] = 0
      g[pos] = 0

      # stag pos
      pos = (row[5], row[4])
      r[pos] = 0.8039
      g[pos] = 0.498
      b[pos] = 0.1961

      # plants pos
      for i in range(6, 12, 2):
          pos = (row[i+1], row[i])
          r[pos] = 0
          b[pos] = 0

      map = np.dstack((r,g,b))
      return map
      
  def valid_action(self, position, action):
    if action == LEFT:
      return position[0] > 0
    elif action == UP:
      return position[1] > 0
    elif action == RIGHT:
      return position[0] < (self.grid_size[1]-1)
    elif action == DOWN:
      return position[1] < (self.grid_size[0]-1)
    return False
  
  def getValidActions(self, position):
    actions = [UP, LEFT, DOWN, RIGHT]
    valid_actions = []
    for a in actions:
      if self.valid_action(position, a):
        valid_actions.append(a)
          
    return valid_actions

  def randomAction(self, state):
    position = (state[0], state[1])
    valid_actions = self.getValidActions(position)
    randomAction = random.choices(valid_actions)[0]
    return randomAction

  def predict_action(self, state):
    state = tf.expand_dims(state, 0)  # Create a batch
    score = self.model.predict(state)[0]
    # action = round(random.random() * self.action_size)

    #fix numeric problem that softmax not always sum to 1
    diff = 1 - sum(score)
    score = score + diff/len(score)
    dict_scores = dict(enumerate(score))
    action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    while(not self.valid_action((state[0][1], state[0][1]), action)):
        del dict_scores[action]
        action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    print(action)
    return action

human_model = HumamModel()
episodes = 5
for ep in range(episodes):    
  obs = env.reset()
  episodes_per_game = 60
  for iteration in range(episodes_per_game):
    env.render()
    computer_action = human_model.randomAction(obs)
    human_action = human_model.predict_action(obs)
    obs, rewards, done, info = env.step([computer_action, human_action])
    obs = obs[0]
    time.sleep(0.5)
    print("info: " ,ep, iteration, rewards, done, info)
env.close()
