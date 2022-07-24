from help_func import *

def calc_distance(pos1, pos2):    
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class ClosestBush:
  def __init__(self, agent, grid_size=(5,5)):
    self.grid_size = grid_size
    self.bushes_pos = []
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
    self.bushes_pos = [[coords_state[i+1], coords_state[i]] for i in range(6,12,2)]

  def takeActionTo(self, current, to):
    if current[0] < to[0]:
        return DOWN #down
    elif current[0] > to[0]:
        return UP #up
    elif current[1] < to[1]:
        return RIGHT #right
    elif current[1] > to[1]:
        return LEFT #left
    return randomAction(self.pos) #takes random action
  
  def act(self, state):
    closest_bush = self.bushes_pos[np.argmin([calc_distance(self.pos, pos2) for pos2 in self.bushes_pos])]
    return self.takeActionTo(self.pos, closest_bush)
