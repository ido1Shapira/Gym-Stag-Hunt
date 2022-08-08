import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import gym

from help_func import *
from monitor import Monitor

from dqn_agent import DQNAgent
from human_model import HumamModel
from follow_stag_agent import FollowStag
from closest_bush_agent import ClosestBush

def run(env, version, prefix_load_human_model, episodes = 1100, epsilon_decay=0.9975, train=False, beta = 0.5, SARL=False,
random_agent=False, follow_stag=False, closest_bush=False, on=None):
  unnormilze_number = 4
  if follow_stag:
    if on=='computer':
        human_model = HumamModel(version, prefix_load_human_model)
        agent_model = FollowStag("computer", on)
        print("follow_stag loaded on computer side")
    else:
        human_model = FollowStag("human", on)
        assert train or random_agent, 'train and random_agent are need to be disabled!'
        agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 0)
        if SARL:
            agent_model.load("data/weights/SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
        else:
            agent_model.load("data/weights/"+prefix_load_human_model+"ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
            episodes = 5
        print("follow_stag loaded on human side")
  elif closest_bush:
    if on=='computer':
        human_model = HumamModel(version, prefix_load_human_model)
        agent_model = ClosestBush("computer", on)
        print("closest_bush loaded on computer side")
    else:
        human_model = ClosestBush("human", on)
        assert train or random_agent, 'train and random_agent are need to be disabled!'
        agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 0)
        if SARL:
            agent_model.load("data/weights/SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
        else:
            agent_model.load("data/weights/"+prefix_load_human_model+"ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
            episodes = 5
        print("closest_bush loaded on human side")
  else:
    human_model = HumamModel(version, prefix_load_human_model)
    # 
    if not train and not random_agent:
        agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 0)
        if SARL:
            agent_model.load("data/weights/SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
        else:
            agent_model.load("data/weights/"+prefix_load_human_model+"ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
            episodes = 5
    else:
        agent_model = DQNAgent((5,5,3), env.action_space.n, epsilon_decay, 1.0)

  monitor = Monitor(version)

  for ep in range(episodes):    
    obs = env.reset()
    if follow_stag or closest_bush:
        if on=='computer':
            agent_model.update_pos(obs)
        else:
            human_model.update_pos(obs)
    if version == 'v2':
        human_model.update_pos(obs)
        agent_model.update_pos(obs)

    obs = vec2mat(obs)
    if not train:
      env.render()
    
    episodes_per_game = 60
    ep_reward = 0
    ep_human_reward = 0
    ep_SARL_reward = 0
    for _ in range(episodes_per_game):
      if random_agent:
        position = np.where(obs[:, :, 2] == np.amax(obs[:, :, 2]))
        computer_action = randomAction(position)
      else:
        binary_action, computer_action = agent_model.act(obs)
      human_action = human_model.act(obs)
      next_obs, rewards, done, info = env.step([computer_action, human_action])
      next_obs = next_obs[0]
      if follow_stag or closest_bush:
        if on=='computer':
            agent_model.update_pos(next_obs)
        else:
            human_model.update_pos(next_obs)
      if version == 'v2':
        human_model.update_pos(next_obs)
        agent_model.update_pos(next_obs)

      next_obs = vec2mat(next_obs)
    #   next_obs = combine_following_states(obs, next_obs)
      agent_reward = rewards[0]
      SARL_reward = beta * agent_reward + (1 - beta) * rewards[1]
      if train:
        if SARL:
          agent_model.remember(np.expand_dims(obs, axis=0), binary_action, SARL_reward, np.expand_dims(next_obs, axis=0), done)
        else:
          agent_model.remember(np.expand_dims(obs, axis=0), binary_action, agent_reward, np.expand_dims(next_obs, axis=0), done)
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
        average = monitor.PlotModel(ep_reward*unnormilze_number, ep_human_reward*unnormilze_number, ep, "SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay))
      else:
        average = monitor.PlotModel(ep_reward*unnormilze_number, ep_human_reward*unnormilze_number, ep, prefix_load_human_model+"ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay))
      print("episode: {}/{}, score: {:.3}, average: {}, e: {:.3}, SARL score: {:.3}".format(ep, episodes, ep_reward*unnormilze_number, average, agent_model.epsilon, ep_SARL_reward*unnormilze_number))
    else:
      print("episode: {}/{}, score: {:.3}, SARL score: {:.3}".format(ep, episodes, ep_reward*unnormilze_number, ep_SARL_reward*unnormilze_number))
    ep_reward = 0
    ep_human_reward = 0
    ep_SARL_reward = 0
    print("counter: {}".format(agent_model.get_bush_counter()))

  if train:
    if SARL:
      agent_model.save("data/weights/SARL_ddqn_agent"+"_"+str(beta)+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
    else:
      agent_model.save("data/weights/"+prefix_load_human_model+"ddqn_agent"+"_"+str(episodes)+"_"+str(epsilon_decay)+"_"+version+".h5")
  env.close()

if __name__ == "__main__":
  env = gym.make("StagHunt-Hunt-v0", obs_type='coords', load_renderer= True, enable_multiagent=True, forage_quantity=3) # you can pass config parameters here

  version = "v2"
  prefix_load_human_model = "" # dropout_ , empathy_
  
  #run random agent
  # run(env, version, prefix_load_human_model, random_agent=True)
  
  #run follow stag agent
  # run(env, version, prefix_load_human_model, follow_stag=True, on='computer')

  #run closest bush agent
#   run(env, version, prefix_load_human_model, closest_bush=True, on='computer')

  #train dqn agent
  # run(env, version, prefix_load_human_model, episodes=4000, epsilon_decay = 0.9995, train=True, SARL=False)
  #test dqn agent
  run(env, version, prefix_load_human_model, episodes=4000, epsilon_decay = 0.9995, train=False, SARL=False)

  #train SARL dqn agent
  # run(env, version, prefix_load_human_model, episodes=4000, epsilon_decay = 0.9995, train=True, beta=0.48 , SARL=True)
  #test SARL dqn agent
  # run(env, version, prefix_load_human_model, episodes=4000, epsilon_decay = 0.9995, train=False, beta=0.2, SARL=True)