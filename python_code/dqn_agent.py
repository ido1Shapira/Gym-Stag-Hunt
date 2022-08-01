import numpy as np
import random

from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

from help_func import *
from follow_stag_agent import FollowStag
from closest_bush_agent import ClosestBush

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, initial_epsilon):
        action_size = 2
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

        self.follow_stag = FollowStag("computer")
        self.closest_bush = ClosestBush("computer")

    def update_pos(self, coords_state):
        self.follow_stag.update_pos(coords_state)
        self.closest_bush.update_pos(coords_state)

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
            binary_action = np.random.choice([0,1])
        else:
            state = tf.expand_dims(state, 0)  # Create a batch
            binary_action = np.argmax(self.model.predict(state))
            
        if binary_action == 0:
            # print("ClosestBushAgent")
            action = self.closest_bush.act(state)
        else:
            # print("FollowStagAgent")
            action = self.follow_stag.act(state)
        return binary_action, action

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
        return self

    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)