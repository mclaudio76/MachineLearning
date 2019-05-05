import random
import numpy as np

import tensorflow as tf
import keras.backend as K
import math,threading,time

from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, SGD
from collections      import deque   
from Commons          import Enviroment,EpisodeStep, History


tensorFlowGraph = None
totalEpisodes   = 0
bestRewardSoFar = None
goal_position   = -0.2



class Mind():
    
    def __init__(self, env_dim, act_dim, policyOptimizer, criticOptimizer):
        self.lock_queue        = threading.Lock()
        self.env_dim           = env_dim
        self.act_dim           = act_dim
        self.__inputNet = self.__base_network()    
        self.__initializePolicyNetwork(policyOptimizer)
        self.__initializeCriticModel(criticOptimizer)
        self.gamma = 0.99
        self.trained = False
        self.episodes = list()
     
    ''' This method builds common shared network,
    responsible for handling inputs
    '''

    def __base_network(self):
        inp       = Input(shape=(self.env_dim,))
        hidden1   = Dense(32,  activation='relu')(inp)
        hidden2   = Dense(64, activation='relu')(hidden1)
        return Model(inp, hidden2)
        
    def __initializePolicyNetwork(self, optimizer):
        hiddenLayer1        = Dense(128, activation='relu') (self.__inputNet.output)
        hiddenLayer2        = Dense(32,  activation='relu') (hiddenLayer1)
        policyOutput        = Dense(self.act_dim, activation='softmax') (hiddenLayer2)
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
        hiddenLayer1        = Dense(64, activation='relu') (self.__inputNet.output)
        hiddenLayer2        = Dense(32, activation='relu') (hiddenLayer1)
        hiddenLayer3        = Dense(12, activation='relu') (hiddenLayer2)
        criticOutput        = Dense(1,   activation='linear') (hiddenLayer3)
        self.__criticModel  = Model(self.__inputNet.input, criticOutput) 
        self.discounted_r   = K.placeholder(shape=(None,))
        loss                = K.mean(K.square(self.discounted_r - self.criticModel().output))
        updates             = optimizer.get_updates(self.criticModel().trainable_weights, [], loss)
        self.__optCritic    = K.function([self.criticModel().input, self.discounted_r], [], updates=updates)


    def pushData(self, episodeData):
        with self.lock_queue:
            self.episodes.append(episodeData)

    def train(self):
        def discount(r):
            discounted_r, cumul_r = np.zeros_like(r), 0
            for t in reversed(range(0, len(r))):
                cumul_r = r[t] + cumul_r * self.gamma
                discounted_r[t] = cumul_r
            return discounted_r
        states  = []
        actions = []
        rewards = []
        with self.lock_queue:
            while len(self.episodes) > 0:
                episodeData = self.episodes.pop()
                for eps in episodeData :
                    states.append(eps.state)
                    actions.append(to_categorical(eps.action, self.act_dim))
                    rewards.append(eps.reward)
                discounted_rewards  = discount(rewards)
                np_state            = np.array(states)
                with tensorFlowGraph.as_default():        
                    state_values        = self.criticModel().predict(np_state)
                    advantages          = discounted_rewards - np.reshape(state_values, len(state_values))
                    #Performs optimization
                    self.__optPolicy([states, actions, advantages])
                    self.__optCritic([states, discounted_rewards])
                self.trained  = True

    def selectAction(self, state):
        def reshapeStateWhenNeeded(state):
            if len(state.shape) < 3: 
                x = np.expand_dims(state, axis=0)
                return x
            return state
        s = reshapeStateWhenNeeded(state)
        with tensorFlowGraph.as_default():
            with self.lock_queue:
                action =  np.random.choice(np.arange(self.act_dim), 1, p=self.policyModel().predict(s).ravel())[0]
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


class Agent():

    def __init__(self, env, mind, rewardMapper=None, epsilon=1.0):
        self.mind              = mind
        self.env               = env
        self.epsilon           = epsilon
        self.epsilon_decay     = 0.9
        self.episodesPlayed    = 0
        self.rewardMapper      = rewardMapper

    def playSingleEpisode(self, render=False):
        global totalEpisodes
        episodeData = list()
        state       = self.env.reset()
        done        = False
        episodeReward = 0
        while not done:
            if np.random.random() < self.epsilon or self.mind.trained == False:
                action =  self.env.randomAction() #self.env.action_space.sample()
            else:
                action = self.mind.selectAction(state)
            next_state, reward, done, __ = self.env.step(action)
            self.env.render(render)
            step                         = EpisodeStep(state, action, reward,next_state, done)
            episodeReward += reward
            episodeData.append(step)
            state = next_state
        if self.rewardMapper != None:
            self.rewardMapper(episodeData)
        totalEpisodes       += 1
        self.mind.pushData(episodeData)
        self.epsilon        *= self.epsilon_decay
        self.epsilon         = 0.0 if self.epsilon < 0.001 else self.epsilon
        return episodeData

class A3C:
   

    class OptimizerThread(threading.Thread):
        stop_signal = False
        def __init__(self, mind):
            self.mind = mind
            threading.Thread.__init__(self)
        def run(self):
            while not self.stop_signal:
                self.mind.train()
        def stop(self):
            self.stop_signal = True

    class AgentThread(threading.Thread):
        stop_signal = False
        def __init__(self, agent, resultHistory):
            threading.Thread.__init__(self)
            self.agent = agent
            self.history = resultHistory
       
        def run(self):
            while not self.stop_signal:
                result = self.agent.playSingleEpisode()
                if result != None:
                    self.history.registerResult(result)

        def stop(self):
            self.stop_signal = True

    def __init__(self, envName, numThreads=3):
        self.lock_queue        = threading.Lock()
        self.enviromentName    = envName
        
        self.env               = Enviroment(envName)
        self.mind              = Mind(self.env.envDim(), self.env.actDim(), policyOptimizer=Adam(lr=0.001), criticOptimizer=Adam(lr=0.0001))
        self.optimizer         = A3C.OptimizerThread(self.mind)
        self.numThreads        = numThreads
        self.results           = History()
        global tensorFlowGraph
        tensorFlowGraph        = K.get_session().graph
        

    def train(self, stopCriterion, numEpisodes = 10000, trainStep=1000, remapRewardFunction=None, epsilon=1.0):
        self.optimizer.start()
        agents = list()
        for __ in range(self.numThreads) :
            agent = Agent(Enviroment(self.enviromentName), self.mind,  rewardMapper=remapRewardFunction, epsilon=epsilon)     
            agThread =  A3C.AgentThread( agent, self.results )
            agThread.start()
            agents.append(agThread)
        done = False
        while not done:
            time.sleep(10)
            done = stopCriterion(list(self.results.results))
        self.optimizer.stop()
        for agent in agents:
            agent.stop()
        self.mind.save(self.enviromentName)

# Examples

def mainCartPole():

    def trainEndedEvaluator(episodeList):
        if len(episodeList) < 100:
            return False
        actualList = episodeList[-100:]
        sumRew = 0
        for episode in actualList:
            for item in episode:
                sumRew += item.reward
        avg = sumRew / len(actualList)
        print(" Average reward  ", avg)
        return avg > 450
   
    a3c = A3C('CartPole-v1',numThreads=4)
    a3c.train(stopCriterion= trainEndedEvaluator)




mainCartPole()

