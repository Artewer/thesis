import numpy as np
import gym
# Libs
import logging
import pandas as pd
from collections import OrderedDict
import pickle
import torch
import gym
from gym import spaces

# Own Modules
from source_torch.mlca.mlca import mlca_mechanism
from source_torch.mlca.mlca_setup import set_value_model_parameters
from source_torch.util import initial_bids_mlca_predefined




class AuctionWorldEnv(gym.Env):

    def __init__(self, configdict = None):
        self.Qinit = configdict['Qinit']
        self.Qmax = configdict['Qmax']
        self.N = 0
        self.M = 0
        if configdict['SATS_domain_name'] == 'LSVM':
            self.N = 6  # number of bidders
            self.M = 18  # number of items
        elif configdict['SATS_domain_n    #Q Algorithm hereame'] == 'GSVM':
            self.N = 7  # number of bidders
            self.M = 18  # number of items
        elif configdict['SATS_domain_name'] == 'MRVM':
            self.N = 10  # number of bidders
            self.M = 98  # number of items
        
        self.bundles_storage, self.fitted_scalar = configdict['init_bids_and_fitted_scaler']
        #create N arrays of M items
        self.bundles_generation = np.zeros((self.N, self.M))
        self.N_tracker = 0
        self.M_tracker = 0

        self.action_space = spaces.Discrete(2)
        self.action_flips = {0:0, 1:1}

    def _get_obs(self):
        return {"bidders": self.N, "bundles": self.bundles, "iter_Qinit": self.iter_Qinit}
    
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.bundles = []
        self.iter_Qinit = 0
        self.bundles_generation = []
        self.N_tracker = 0
        self.M_tracker = 0

        observation = self._get_obs()

        return observation
    
    def step(self, action):
        terminated = False

        if self.N_tracker == self.N-1:
            pass
        else:
            if self.M_tracker == self.M-1:
                self.N_tracker += 1
                self.M_tracker = 0
            else:
                self.bundles_generation[self.N_tracker][self.M_tracker] = action
                self.M_tracker += 1

        self.bundles_storage.append(self.bundles_generation)
        
        self.iter_Qinit += 1

        configdict_mlca = set_value_model_parameters(configdict_mlca)
        res = mlca_mechanism(configdict = configdict_mlca)
        try:            
            scores = res['MLCA Efficiency']
        except:
            scores = 0

        # An episode is done iff the agent has Qinit items
        if self.iter_Qinit == self.Qinit:
            terminated = True

        reward = float(scores)

        observation = self._get_obs()
        
        return observation, reward, terminated, scores
    

def q_learning(env, num_episodes, learning_rate, discount_factor, epsilon):

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()  # Choose a random action
            else:
                action = np.argmax(Q[state])  # Choose the best action based on current Q-values

            next_state, reward, done, _ = env.step(action)

            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q[next_state])
            )

            state = next_state

    return Q