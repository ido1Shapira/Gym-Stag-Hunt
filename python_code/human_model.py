from help_func import *
import tensorflow as tf

from closest_bush_agent import ClosestBush
from follow_stag_agent import FollowStag

class HumamModel:
  def __init__(self, version, prefix_load_human_model):
    self.model = tf.keras.models.load_model('./data/humanModel/'+prefix_load_human_model+'model_'+version+'.h5')
    self.version = version

    if self.version == "v2":
      self.follow_stag = FollowStag("human")
      self.closest_bush = ClosestBush("human")
  
  def update_pos(self, coords_state):
    self.follow_stag.update_pos(coords_state)
    self.closest_bush.update_pos(coords_state)
    
  def act(self, state):
    state = tf.expand_dims(state, 0)  # Create a batch
    score = self.model.predict(state)[0]

    #fix numeric problem that softmax not always sum to 1
    diff = 1 - sum(score)
    score = score + diff/len(score)
    dict_scores = dict(enumerate(score))
    action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    
    if self.version != "v2":
      position = np.where(state[:, :, 0] == np.amax(state[:, :, 0]))
      while(not valid_action(position, action, (5,5))):
          del dict_scores[action]
          action = random.choices(list(dict_scores.keys()), weights=list(dict_scores.values()))[0]
    else:
      if action == 0:
        print("FollowStagAgent")
        action = self.follow_stag.act(state)
      else:
        print("ClosestBushAgent")
        action = self.closest_bush.act(state)
    return action