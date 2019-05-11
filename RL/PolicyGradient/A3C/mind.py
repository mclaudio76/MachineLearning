import random
import numpy as np
import gym
import commons
import tensorflow as tf
import keras.backend as K
import math,threading,time
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten,LSTM
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
 

class Mind():
    
    def __init__(self, env_dim, act_dim, policyOptimizer, criticOptimizer,trained = False):
        self.lock_queue        = threading.Lock()
        self.env_dim           = env_dim
        self.act_dim           = act_dim
        self.__inputNet = self.__base_network()    
        self.__initializePolicyNetwork(policyOptimizer)
        self.__initializeCriticModel(criticOptimizer)
        self.gamma = 0.99
        self.trained = trained
        self.episodes = list()
     
    ''' This method builds common shared network,
    responsible for handling inputs
    '''

    def __base_network(self):
        inp       = Input(shape=(self.env_dim,))
      
        hidden1   = Dense(12,  activation='relu')(inp)
        hidden2   = Dense(24, activation='relu')(hidden1)
        return Model(inp, hidden2)
        
    def __initializePolicyNetwork(self, optimizer):
        hiddenLayer1        = Dense(12, activation='relu') (self.__inputNet.output)
        policyOutput        = Dense(self.act_dim, activation='softmax') (hiddenLayer1)
        self.__policyModel  = Model(self.__inputNet.input, policyOutput)
        
        #points to input tensor
        self.action_pl      = K.placeholder(shape=(None, self.act_dim))
        #points to advantages, i.e discounted reward.
        self.advantages_pl  = K.placeholder(shape=(None,))
        
        #network output is multiplied by action input, i.e  a tensor of one-hot-encoded actions.
        actions_probabilities    = K.sum(self.action_pl * self.policyModel().output, axis=1)
        # multiply log of probabilities (to which is added a constant, small term to avoid 0-values),
        # for the value assumed by the advantages vector (K.stop_gradient is needed to treat the tensor as a constant)
        action_value             = K.log(actions_probabilities + 1e-10) * K.stop_gradient(self.advantages_pl)
        
        # Adding an "entropy" term, it should encourage exploration of new actions
        entropy                  = K.sum(self.policyModel().output * K.log(self.policyModel().output + 1e-10), axis=1)

        # Defining a loss.
        loss                     = 0.001 * entropy - K.sum(action_value)
      
        updates                  = optimizer.get_updates(self.policyModel().trainable_weights, [], loss)
        
        self.__optPolicy        = K.function([self.policyModel().input, self.action_pl, self.advantages_pl], [], updates=updates)
    
    def __initializeCriticModel(self, optimizer):
        hiddenLayer1        = Dense(12, activation='relu') (self.__inputNet.output)
        criticOutput        = Dense(1,   activation='linear') (hiddenLayer1)
        self.__criticModel  = Model(self.__inputNet.input, criticOutput) 
        self.discounted_r   = K.placeholder(shape=(None,))
        loss                = K.mean(K.square(self.discounted_r - self.criticModel().output))
        updates             = optimizer.get_updates(self.criticModel().trainable_weights, [], loss)
        self.__optCritic    = K.function([self.criticModel().input, self.discounted_r], [], updates=updates)


    def pushData(self, episodeData):
        with self.lock_queue:
            self.episodes.append(episodeData)
    

    def train(self):
        states  = []
        actions = []
        rewards = [] 
        with self.lock_queue:
            while len(self.episodes) > 0:
                episodeData = self.episodes.pop()
                for eps in episodeData :
                    states.append(eps.state)
                    act = to_categorical(eps.action, self.act_dim,dtype='int32')
                    actions.append(act)
                    rewards.append(eps.reward)
                np_state            = np.array(states)
                with commons.tensorFlowGraph.as_default():        
                    state_values        = self.criticModel().predict(np_state)
                    advantages          = rewards - np.reshape(state_values, len(state_values))
                    #Performs optimization
                    self.__optPolicy([states, actions, advantages])
                    self.__optCritic([states, rewards])
                self.trained  = True

    def selectAction(self, state):
        def reshapeStateWhenNeeded(state):
            if len(state.shape) < 3: 
                x = np.expand_dims(state, axis=0)
                return x
            return state
        s = reshapeStateWhenNeeded(state)
        with commons.tensorFlowGraph.as_default():
            with self.lock_queue:
                #action =  np.random.choice(np.arange(self.act_dim), 1, p=self.policyModel().predict(s).ravel())[0]
                p=self.policyModel().predict(s).ravel()
                action = np.argmax(p)
                return action

    def policyModel(self):
        return self.__policyModel
    
    def criticModel(self):
        return self.__criticModel

    def save(self, path):
        self.policyModel().save_weights(path+"-policy.model")
        self.criticModel().save_weights(path+"-critic.model")

    def load_weights(self, path):
        self.policyModel().load_weights(path+"-policy.model")
        self.criticModel().load_weights(path+"-critic.model")
