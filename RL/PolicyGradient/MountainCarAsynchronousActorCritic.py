import random
import numpy as np
import gym
import tensorflow as tf
import keras.backend as K
import math,threading,time
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten,LSTM
from keras.optimizers import Adam, SGD
from Commons          import Enviroment,EpisodeStep, History
import matplotlib.pyplot as plt
 
tensorFlowGraph = None
totalEpisodes   = 0
bestRewardSoFar = None
goal_position   = 0.5



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
                with tensorFlowGraph.as_default():        
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
        with tensorFlowGraph.as_default():
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


class Agent():

    def __init__(self, env, mind, rewardMapper=None, epsilon=1.0, name=None):
        self.mind              = mind
        self.env               = env
        self.name              = name
        self.epsilon           = epsilon
        self.epsilon_decay     = 0.995
        self.episodesPlayed    = 0
        self.rewardMapper      = rewardMapper
        self.agentEpisode      = 0
        self.gamma             = 0.99

    def discount(self, episodeData):
        discounted_r, cumul_r = np.zeros_like(episodeData), 0
        for t in reversed(range(0, len(episodeData))):
            episodestep = episodeData[t]
            cumul_r = episodestep.reward + cumul_r * self.gamma
            episodestep.reward = cumul_r
        

    def playSingleEpisode(self, render=False):
        global totalEpisodes
        episodeData = list()
        state       = self.env.reset()
        done        = False
       # render      = self.name == "Agent-0" and self.agentEpisode  % 10 == 0
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
        
        totReward = 0
        if self.rewardMapper != None:
            totReward = self.rewardMapper(episodeData)
        totalEpisodes       += 1
        self.agentEpisode   += 1
        self.discount(episodeData)
        self.mind.pushData(episodeData)
        self.epsilon        *= self.epsilon_decay
        self.epsilon         = 0.01 if self.epsilon < 0.01 else self.epsilon
        return episodeData
        
    
    def goodEpisode(self, episodeData):
        return True

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
        self.mind              = Mind(self.env.envDim(), self.env.actDim(), policyOptimizer=Adam(lr=0.001), criticOptimizer=Adam(lr=0.001))
        self.optimizer         = A3C.OptimizerThread(self.mind)
        self.numThreads        = numThreads
        self.results           = History()
        global tensorFlowGraph
        tensorFlowGraph        = K.get_session().graph
        

    def train(self, stopCriterion, numEpisodes = 10000, trainStep=1000, remapRewardFunction=None, epsilon=1.0):
        self.optimizer.start()
        agents = list()
        for idEnv in range(self.numThreads) :
            agent = Agent(Enviroment(self.enviromentName, seed=idEnv), self.mind,  rewardMapper=remapRewardFunction, epsilon=epsilon, name="Agent-"+str(idEnv))     
            agThread =  A3C.AgentThread( agent, self.results )
            agThread.start()
            agents.append(agThread)
        done = False
        while not done:
            time.sleep(1)
            done = stopCriterion(list(self.results.results))
        self.optimizer.stop()
        for agent in agents:
            agent.stop()
        self.mind.save(self.enviromentName)

# Examples

def mainMountain():

    def trainEndedEvaluator(episodeList):
        if len(episodeList) < 10:
            return False
        global totalEpisodes
        global goal_position
        global bestRewardSoFar
        actualList = episodeList[-10:]
        sumRew = 0
        solved = 0
        minP    = 100
        maxP    = -100
        velocity = 0.0
        changes    = 0
        for episode in actualList:
            lastAction = None
            for item in episode:
                if lastAction != item.action:
                    lastAction = item.action
                    changes +=1
                sumRew += item.reward
                position = item.next_state[0]
                velocity += item.next_state[1]
                minP     = min(minP, position)
                maxP     = max(maxP, position)
                if position >= goal_position:
                    solved += 1
        avg = sumRew / len(actualList)
        avgChanges = changes / len(actualList)
        avgSpeed = velocity / (len(actualList) * 200)
        bestRewardSoFar = avg
        print("Avg Changes", avgChanges, " Average reward  after ", totalEpisodes, " episodes is ", avg, "Solved = ", solved, " best range (",minP, ", ", maxP,") , avg speed ",avgSpeed)
        if solved / 10.0 > 0.9:
            return True
        
    
    def remapReward(episodeData):
        x0          = -0.5 
        global goal_position
        totalReward  = 0
        bestPosition = -2
        minPos  = +5
        maxPos  = -5
        for eps in episodeData:
            position   = eps.next_state[0]
            minPos     = min(minPos, position)
            maxPos     = max(maxPos, position)
        delta_inf      = x0-minPos
        delta_sup      = maxPos - x0
        rew            = abs(delta_sup) - abs(delta_inf)
        for eps in episodeData:
            eps.reward = rew / len(episodeData)
        totalReward = rew
        return totalReward
            
    a3c = A3C('MountainCar-v0',numThreads=10)
    a3c.train(stopCriterion= trainEndedEvaluator,remapRewardFunction=remapReward, epsilon=1.0)


mainMountain()

