from help_func import *

class FollowStag:
  def __init__(self, agent, grid_size=(5,5)):
    self.grid_size = grid_size
    self.stag_pos = [2,2]
    self.agent = agent
    
    if self.agent == "computer":
        self.pos = [0,4]
    else: # self.agent == "human"
        self.pos = [0,0]
  
  def update_pos(self, coords_state):
    if self.agent == "computer":
        self.pos = [coords_state[1], coords_state[0]]
    else: # self.agent == "human"
        self.pos = [coords_state[3], coords_state[2]]
    self.stag_pos = [coords_state[5], coords_state[4]]

  def takeActionTo(self):
    if self.pos[0] < self.stag_pos[0]:
        return DOWN #down
    elif self.pos[0] > self.stag_pos[0]:
        return UP #up
    elif self.pos[1] < self.stag_pos[1]:
        return RIGHT #right
    elif self.pos[1] > self.stag_pos[1]:
        return LEFT #left
    return randomAction(self.pos) #takes random action
  
  def act(self, state):
    return self.takeActionTo()