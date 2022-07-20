from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND

import numpy as np
import random

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
  r[coords_state[5], coords_state[4]] += 0.5
  g[coords_state[5], coords_state[4]] += 0.5
  b[coords_state[5], coords_state[4]] += 0.5
  # plants pos
  for i in range(6, 12, 2):
      g[coords_state[i+1], coords_state[i]] += 1

  return np.dstack((r,g,b))

def combine_following_states(prev, current):
  r2, g2, b2 = current[:, :, 0], current[:, :, 1], current[:, :, 2]
  human_pos = np.where((r2 == 1) | (r2 == 1.5))
  computer_pos = np.where((b2 == 1) | (b2 == 1.5))
  bushes_pos = np.where((g2 == 1) | (g2 == 1.5))
  stag_pos = np.where(((r2 == 0.5) & (g2 == 0.5) & (b2 == 0.5)) |
                      ((r2 == 1.5) & (g2 == 0.5) & (b2 == 0.5)) |
                      ((r2 == 0.5) & (g2 == 1.5) & (b2 == 0.5)) |
                      ((r2 == 0.5) & (g2 == 0.5) & (b2 == 1.5)))

  new_cell = prev * 0.9

  new_cell[:, :, 0][human_pos] = 1
  new_cell[:, :, 1] = np.zeros([5,5])
  new_cell[:, :, 1][bushes_pos] = 1
  new_cell[:, :, 2][computer_pos] = 1

  new_cell[:, :, 0][stag_pos] += 0.5
  new_cell[:, :, 1][stag_pos] += 0.5
  new_cell[:, :, 2][stag_pos] += 0.5

  return NormalizeData(new_cell)