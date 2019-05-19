import gym
import random
import math
import sys
import numpy as np
import tensorflow as tf

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam, SGD
from keras.models     import load_model
from collections      import deque
from keras            import backend as K


'''
Mauri Claudio, 25.04.2019

Implementation of a DeepQ-Learning based Agent.
Supports both DQ and DDQ learning approaches
'''

class Memory: 
    samples = []
    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        
        if self.isFull():
            self.samples.pop(0)
    
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
    def isFull(self):
        return len(self.samples) >= self.capacity


class EpisodeStep:
    def __init__(self, state, action, reward, next_state, done):
        self.state      = state
        self.action     = action
        self.reward     = reward
        self.next_state = next_state
        self.done       = done

class Mind:

    def __init__(self, env, epsilon_max = 1.0, epsilon_min=0.01, epsilon_decay=0.001, gamma_reward = 0.95, DDQEnabled=True, DQQSwap=1000, memorySize=100000):
        self.env               = env
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n 
        
        # epsilon parameters for exploring space
        self.epsilon_max       = epsilon_max
        self.epsilon_decay     = epsilon_decay
        self.epsilon_min       = epsilon_min            
        self.epsilon           = epsilon_max
        
        self.gamma             = gamma_reward    # how much future rewards are worth?
        self.DDQEnabled        = DDQEnabled      # enables DDQ Learning
        self.DQQSwap           = DQQSwap
        self.memory            = Memory(capacity=memorySize)
        # in DDQ learning, two distinc models are created.
        self.qfunct_current    = self.__buildModel(self.state_size, self.action_size)
        self.qfunct_target     = self.__buildModel(self.state_size, self.action_size)
        self.train_runs        = 0

    def loadModels(self, enviromentName):
        prefix = "DDQ" if self.DDQEnabled else "DQ"
        self.qfunct_current.load_weights(prefix+enviromentName+"-agent.hdf5")
        if self.DDQEnabled:
            self.qfunct_target.load_weights(prefix+enviromentName+"-qvalue.hdf5")
        self.epsilon        = 0.00

    def saveModels(self, enviromentName):
        prefix = "DDQ" if self.DDQEnabled else "DQ"
        self.qfunct_current.save_weights(prefix+enviromentName+"-agent.hdf5")
        if self.DDQEnabled:
            self.qfunct_target.save_weights(prefix+enviromentName+"-qvalue.hdf5")

    '''
    Builds a NN model.
    In DQ learning, a single NN is used both to choose actions to perform 
    and to approxymate the Q-Value(state, action) function.
    In DDQ Learning, to increase learning stability, two distict NN are used: the first one
    chooses the action to perform (given a state), while the other one approxymates Q-Value function.
    Every n steps the Q-Value network's (also known as 'target' network) weights are updated with
    the 'current' network weights
    '''
    def __buildModel(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        #model.add(Dense(12, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        optimizer = Adam(lr=0.001) 
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def __computeQValue(self, model, state):
        input    = state.reshape(-1,self.state_size)
        rewards  = model.predict(input)[0]
        return rewards

    '''
    Choose which action to perform. Please note that in both DQ and DDQ approaches,
    is always used the "current" network to compute Q-Values used to choose best action.
    The policy followed  by the agent as a whole is always greedy, i.e it's always choosen 
    the action with  higher Q value (as extimated by the NN)
    '''
    def selectAction(self,state):
        if np.random.random() <= self.epsilon :
            action =  self.env.action_space.sample()
        else:
            action = np.argmax(self.__computeQValue(self.qfunct_current,state))
        return action

    def memorize(self, episodestep):
        self.memory.add(episodestep)

    '''
    Trains the NN using experience replay. A sigle step is added to the memory
    buffer, from wich a small sample is taken to train the NN.
    '''
    def train(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        self.__batch_train(minibatch)
        self.train_runs += 1
        self.epsilon  = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-self.epsilon_decay * self.train_runs)
        if self.train_runs > self.DQQSwap and self.train_runs % self.DQQSwap == 0 and self.DDQEnabled:    
            self.qfunct_target.set_weights(self.qfunct_current.get_weights())
        

    '''
    Actual train code. A subset of the whole episode history is passed to this method.
    If the dataset is empty no training is performed.
    Otherwise, we'll use the "current" or the "target" model to compute updates
    for the Q- function, using Bellman's equation.
    '''
    def __batch_train(self, dataset):
        if len(dataset) == 0:
            return
        state_array, Qvalues_array = [], []
        q_update_model = self.qfunct_target if self.DDQEnabled  else self.qfunct_current
        for episodeStep in dataset:
            q_update     = 0
            # This line of code computes Q(currentState, a) for each a in A, in a single step.
            q_values     = self.__computeQValue(self.qfunct_current,episodeStep.state)
            # If we're not in a terminal state:
            if not episodeStep.done:
                # Compute max value for the next state, using 'agent' or 'predict' model 
                # when using DQ or DDQ, respectively.
                q_update = np.amax(self.__computeQValue(q_update_model,episodeStep.next_state)) 
                # q_update takes in account discarded future rewards.
                q_update = episodeStep.reward + self.gamma * q_update
            else:
                q_update = episodeStep.reward
            # we prepare two distinct arrays to train the NN ()
            q_values[episodeStep.action] = q_update
            state_array.append(episodeStep.state)
            Qvalues_array.append(q_values)
        
        X = np.array([x for x in state_array]).reshape(-1, self.state_size)
        y = np.array([y for y in Qvalues_array]).reshape(-1, self.action_size)
        self.qfunct_current.fit(X, y, epochs=1, verbose=0)



class DQAgent:
    
    def __init__(self, envName, enviroment, mind, train = True, custom_reward = None):
        self.mind    = mind
        self.env     = enviroment
        self.envName = envName
        self.train   = train
        self.custom_reward = custom_reward
        if not self.train:
            mind.loadModels(envName)
       
    '''
    Collects and returns the whole episode steps' data.
    Each episode step is pushed to the memory of the 'mind'
    and a train sequence is performed
    '''
    def playSingleEpisode(self):
        episodeData = list()
        state       = self.env.reset()
        done        = False
        while not done:
            action                       = self.mind.selectAction(state)
            next_state, reward, done, __ = self.env.step(action)
            if not self.train:
                self.env.render()
            step                         = EpisodeStep(state, action, reward,next_state, done)
            if self.custom_reward != None:
                self.custom_reward(step)
            episodeData.append(step)
            self.mind.memorize(step)
            state = next_state
            if self.train:
                self.mind.train(64)
        return episodeData



#Training CartPole

def trainCartPole():
    
    def totalReward(episodeData):
        totalReward = 0
        for epsStep in episodeData:
            totalReward += epsStep.reward
        return totalReward
    
    def finished(results):
        LIMIT = 100
        if len(results) < LIMIT:
            return False
        sample = results[-LIMIT:]
        tReward = 0
        for x in range(LIMIT):
            tReward += totalReward(sample[x])
        return tReward / LIMIT >= 450

    results = list()
    episodeNumber     = 100000
    enviromentName    = "CartPole-v1"
    env = gym.make(enviromentName) # creates enviroment using symbolic name
    env.reset()                    # reset enviroment
    mind = Mind(env)
    agent = DQAgent(enviromentName,env,mind)
    for currentEpisode in range(episodeNumber):
        episodeData = agent.playSingleEpisode()
        reward      = totalReward(episodeData)
        if currentEpisode % 50 == 0:
            print("Playing current episode ", currentEpisode, " reward ", reward)
        results.append(episodeData)
        if finished(results):
            mind.saveModels(enviromentName)
            print("Train successful !!! ")
            return

def playCartPole():
    
    def totalReward(episodeData):
        totalReward = 0
        for epsStep in episodeData:
            totalReward += epsStep.reward
        return totalReward
    episodeNumber     = 10
    enviromentName    = "CartPole-v1"
    env = gym.make(enviromentName) # creates enviroment using symbolic name
    env.reset()                    # reset enviroment
    mind = Mind(env)
    agent = DQAgent(enviromentName,env,mind, train=False)
    for currentEpisode in range(episodeNumber):
        episodeData = agent.playSingleEpisode()
        reward      = totalReward(episodeData)
        print("Playing current episode ", currentEpisode, " reward ", reward)
        
        
#trainCartPole()     
playCartPole()
    