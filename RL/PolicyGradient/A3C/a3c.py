import random
import numpy as np
import gym
import commons
import tensorflow as tf
import keras.backend as K
import math,threading,time

from  mind import Mind
from  agent import Agent
from  commons import EpisodeStep, History, Enviroment
from keras.optimizers import Adam, RMSprop


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

        commons.tensorFlowGraph   = K.get_session().graph
        

    def train(self, stopFunction, remapRewardFunction=None, epsilon=1.0):
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
            done = stopFunction(list(self.results.results))
        self.optimizer.stop()
        for agent in agents:
            agent.stop()
        self.mind.save(self.enviromentName)

    def play(self):
        self.mind.trained = True
        self.mind.load_weights(self.enviromentName)
        agent = Agent(Enviroment(self.enviromentName), self.mind, epsilon=0.0)  
        return agent.playSingleEpisode(render=True) 