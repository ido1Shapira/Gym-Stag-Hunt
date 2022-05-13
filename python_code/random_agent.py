import gym
import gym_stag_hunt
import time

env = gym.make("StagHunt-Hunt-v0", obs_type='image', enable_multiagent=True, grid_size=(5,5)) # you can pass config parameters here

episodes = 5
for ep in range(episodes):    
  env.reset()
  episodes_per_game = 100
  for iteration in range(episodes_per_game):
    time.sleep(.2)
    obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
    env.render()

    print("info: " ,ep, iteration, rewards, done, info)
env.close()
