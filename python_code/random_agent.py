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

class HumamModel:
  def __init__(self):
    self.model = tf.keras.models.load_model('./data/humanModel/model_v0.h5')
    self.grid_size = (5,5)
  def predict_action(self, state):
    position = np.where(state[:, :, 0] == np.amax(state[:, :, 0]))
    state = tf.expand_dims(state, 0)  # Create a batch
    score = self.model.predict(state)[0]
    # action = round(random.random() * self.action_size)

    #fix numeric problem that softmax not always sum to 1
    diff = 1 - sum(score)
    score = score + diff/len(score)
    dict_scores = dict(enumerate(score))
    action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    while(not self.valid_action(position, action)):
        del dict_scores[action]
        action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    return action
      
  def valid_action(self, position, action):
    if action == LEFT:
      return position[1] > 0
    elif action == UP:
      return position[0] > 0
    elif action == RIGHT:
      return position[1] < (self.grid_size[1]-1)
    elif action == DOWN:
      return position[0] < (self.grid_size[0]-1)
    return False
  
  def getValidActions(self, position):
    actions = [UP, LEFT, DOWN, RIGHT]
    valid_actions = []
    for a in actions:
      if self.valid_action(position, a):
        valid_actions.append(a)
          
    return valid_actions

  def randomAction(self, state):
    position = np.where(state[:, :, 2] == np.amax(state[:, :, 2]))
    valid_actions = self.getValidActions(position)
    randomAction = random.choices(valid_actions)[0]
    return randomAction

  def predict_action(self, state):
    position = np.where(state[:, :, 0] == np.amax(state[:, :, 0]))
    state = tf.expand_dims(state, 0)  # Create a batch
    score = self.model.predict(state)[0]
    # action = round(random.random() * self.action_size)

    #fix numeric problem that softmax not always sum to 1
    diff = 1 - sum(score)
    score = score + diff/len(score)
    dict_scores = dict(enumerate(score))
    action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    while(not self.valid_action(position, action)):
        del dict_scores[action]
        action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    return action

human_model = HumamModel()
episodes = 5
for ep in range(episodes):    
  obs = env.reset()
  obs = vec2mat(obs)
  episodes_per_game = 60
  for iteration in range(episodes_per_game):
    env.render()
    computer_action = human_model.randomAction(obs)
    human_action = human_model.predict_action(obs)
    next_obs, rewards, done, info = env.step([computer_action, human_action])
    next_obs = next_obs[0]
    next_obs = vec2mat(next_obs)
    # next_obs = combine_following_states(obs, next_obs)
    time.sleep(0.2)
    obs = next_obs
    print("info: " ,ep, iteration, rewards, done, info)
env.close()
