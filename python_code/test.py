import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import gym

from help_func import *

from dqn_agent import DQNAgent
from human_model import HumamModel
from follow_stag_agent import FollowStag
from closest_bush_agent import ClosestBush

def test(env, computer, human, episodes):
  computer_scores, human_scores = [],[]

  for ep in range(episodes):    
    obs = env.reset()
    
    computer.update_pos(obs)
    human.update_pos(obs)

    obs = vec2mat(obs)
    
    ep_reward = 0
    ep_human_reward = 0
    for _ in range(60):
      
      _, computer_action = computer.act(obs)
      human_action = human.act(obs)
      next_obs, rewards, _, _ = env.step([computer_action, human_action])
      next_obs = next_obs[0]
      
      computer.update_pos(next_obs)
      human.update_pos(next_obs)

      next_obs = vec2mat(next_obs)
 
      obs = next_obs
      ep_reward += rewards[0]
      ep_human_reward += rewards[1]      
    
    # print("episode: {}/{}, score: {:.3}".format(ep, episodes, ep_reward*5))

    computer_scores.append(ep_reward*5)
    human_scores.append(ep_human_reward*5)

    ep_reward = 0
    ep_human_reward = 0
  
  return sum(computer_scores) / len(computer_scores), sum(human_scores) / len(human_scores)


if __name__ == "__main__":
  env = gym.make("StagHunt-Hunt-v0", obs_type='coords', load_renderer= True, enable_multiagent=True, forage_quantity=3) # you can pass config parameters here

  computer_agents = {'sarl ddqn 0.2': DQNAgent((5,5,3), env.action_space.n, 0.9995, 0).load("data/weights/SARL_ddqn_agent_0.2_4000_0.9995_v2.h5"),
                     'ddqn': DQNAgent((5,5,3), env.action_space.n, 0.999, 0).load("data/weights/ddqn_agent_4000_0.9995_v2.h5"),
                    #  'sarl ddqn 0.48': DQNAgent((5,5,3), env.action_space.n, 0.9995, 0).load("data/weights/SARL_ddqn_agent_0.48_4000_0.9995_v2.h5")
                    }
  
  version = "v2"
  prefix_load_human_model = "" # dropout_ , empathy_
  human_agents = {'follow_stag': FollowStag("human"),
                  'closest_bush': ClosestBush("human"),
                  'human_model': HumamModel(version, prefix_load_human_model),
                }

  episodes = 250
  
  for agent in computer_agents:
    computer_avg, human_avg = [], []
    for human in human_agents:
      start_time = time.time()
      computer_average, human_average = test(env, computer_agents[agent], human_agents[human], episodes)
      computer_avg.append(computer_average)
      human_avg.append(human_average)
      print('________________________________________________________________________________')
      print(agent + ' | ' + human + ' | computer_average='+ str(computer_average) + '| human_average=' + str(human_average) + ' | time=' + str(time.time() - start_time))
    print('________________________________________________________________________________')
    print(agent + ' | computer_average='+ str(computer_average) + '| human_average=' + str(human_average))
    print('________________________________________________________________________________')
