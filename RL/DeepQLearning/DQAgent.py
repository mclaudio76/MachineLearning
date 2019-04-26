import gym
import random
import sys
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam, SGD
from keras.models     import load_model
from collections      import deque 
'''
Mauri Claudio, 25.04.2019

Implementation of a DeepQ-Learning based Agent.
Supports both DQ and DDQ learning approaches
'''

class EpisodeStep:
    def __init__(self, state, action, reward, next_state, done):
        self.state      = state
        self.action     = action
        self.reward     = reward
        self.next_state = next_state
        self.done       = done

class DQAgent:
    
    def __init__(self, enviromentName: str, memoryBufferSize: int, epsilon = 1.0, epsilon_decay=0.95, discount_factor = 0.95, train=True, DDQEnabled=False):
        
        self.enviromentName    = enviromentName
        self.env = gym.make(enviromentName) # creates enviroment using symbolic name
        self.env.reset()                    # reset enviroments
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n 
        
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        
        self.gamma              = discount_factor # how much a future reward is worth ?
        self.DDQEnabled         = DDQEnabled      # enables DDQ Learning
       
        self.last_performance_index  = 0
        self.best_performance_index  = 0
       
        
        self.memory         = deque(maxlen=memoryBufferSize)

        self.agent_model        = self.__buildModel(self.state_size, self.action_size)
        self.qfunction_model    = self.__buildModel(self.state_size, self.action_size)

        self.trainingMode   = train

        if not self.trainingMode:
            self.loadModels()
  
    '''
    Builds two distinct, simmetric, neural networks. In DQ learning, we can use a single
    NN  both to choose actions to perform and to approxymate Q-Value(state, action) function.
    In DDQ Learning, to increase learning stability, two distict NN are used: the first one
    chooses the action to perform (given a state), while the other one approxymates Q-Value function.
    Every n steps the Q-Value network's (also known as 'target' network) weights are updated with
    the 'agent' network.
    '''
    def __buildModel(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(12, input_dim=input_size, activation='relu'))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        optimizer = Adam(lr=0.0001) 
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    def computeQValue(self, model, state):
        input    = state.reshape(-1,self.state_size)
        rewards  = model.predict(input)[0]
        return rewards

    
    def loadModels(self):
        prefix = "DDQ" if self.DDQEnabled else "DQ"
        self.agent_model.load_weights(prefix+self.enviromentName+"-agent.hdf5")
        if self.DDQEnabled:
            self.qfunction_model.load_weights(prefix+self.enviromentName+"-qvalue.hdf5")
        self.epsilon        = 0.00

    def saveModels(self):
        prefix = "DDQ" if self.DDQEnabled else "DQ"
        self.agent_model.save_weights(prefix+self.enviromentName+"-agent.hdf5")
        if self.DDQEnabled:
            self.qfunction_model.save_weights(prefix+self.enviromentName+"-qvalue.hdf5")

    '''
    Choose which action to perform. Please note that in both DQ and DDQ approaches,
    is always the "agent" network that is used to compute Q-Values. The policy followed
    by the agent as a whole is always greedy, i.e it's always choosen the action which
    has the higher Q value (as extimated by the NN)
    '''
    def selectAction(self,state):
        if self.trainingMode and (np.random.random() <= self.epsilon) :
            action =  self.env.action_space.sample()
        else:
            action = np.argmax(self.computeQValue(self.agent_model,state))
        return action


    '''
    Actual train code. A subset of the whole train history is passed to this method.
    If the dataset is empty no training is performed.
    Otherwise, we'll use the "agent" or the "predict" model to compute updates
    for the Q- function, using Bellman's equation.
    
    '''
    def batch_train(self, dataset):
        if len(dataset) == 0:
            return
        state_array, Qvalues_array = [], []
        q_update_model = self.qfunction_model if self.DDQEnabled  else self.agent_model
        for episodeStep in dataset:
            q_update     = 0
            # This line of code computes Q(currentState, a) for each a in A, in a single step.
            q_values     = self.computeQValue(self.agent_model,episodeStep.state)
            # If we're not in a terminal state:
            if not episodeStep.done:
                # Compute max value for the next state, using 'agent' or 'predict' model 
                # when using DQ or DDQ, respectively.
                q_update = np.amax(self.computeQValue(q_update_model,episodeStep.next_state)) 
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
        self.agent_model.fit(X, y, epochs=1, steps_per_epoch=100,verbose=0)


    '''
    Collects and returns the whole episode steps' data.
    '''
    def playSingleEpisode(self, render=False):
        episodeData = list()
        state       = self.env.reset()
        done        = False
        while not done:
            action                       = self.selectAction(state)
            next_state, reward, done, __ = self.env.step(action)
            if render:
                self.env.render()
            step                         = EpisodeStep(state, action, reward,next_state, done)
            episodeData.append(step)
            state = next_state
        return episodeData


    def pushToMemory(self, episodeData):      
        for x in episodeData:
            self.memory.appendleft(x)

    def train(self, stopFunction, trainStep = 100, swapStep  = 100, numEpisodes=10000, minibatchSize=1000, customize_reward_function = None, progress_log_funct = None):
        results = list()
        for currentEpisode in range(numEpisodes):
            episodeData = self.playSingleEpisode()
            if customize_reward_function != None:
                customize_reward_function(episodeData)
            results.append(episodeData)
            self.pushToMemory(episodeData)        
            if stopFunction(results):
                self.saveModels()
                print ( " Train ended.")
                return
            else:
                minibatch = random.sample(self.memory, min(len(self.memory), minibatchSize))
                self.batch_train(minibatch)
            if currentEpisode > 0 and currentEpisode % trainStep == 0:
                print("[Episode ", currentEpisode, "/",numEpisodes,"] ",end='\n' if progress_log_funct==None else ' ')
                self.saveModels()
                if progress_log_funct != None:
                   progress_log_funct(results)
                self.epsilon *= self.epsilon_decay
                self.epsilon  = 0 if self.epsilon < 0.001 else self.epsilon
            if currentEpisode > 0 and currentEpisode % swapStep == 0:    
                self.qfunction_model.set_weights(self.agent_model.get_weights())
        print( "Train failed, no stop condition met.")

    