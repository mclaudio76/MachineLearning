import random
import numpy as np
import commons
import math,threading,time
from commons import EpisodeStep,Enviroment


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

    def discountRewards(self, episodeData):
        cumul_r = 0
        for t in reversed(range(0, len(episodeData))):
            episodestep = episodeData[t]
            cumul_r = episodestep.reward + cumul_r * self.gamma
            episodestep.reward = cumul_r

    def playSingleEpisode(self, render=False):
        episodeData = list()
        state       = self.env.reset()
        done        = False
        episodeReward = 0
        while not done:
            if np.random.random() < self.epsilon or self.mind.trained == False:
                action =  self.env.randomAction() 
            else:
                action = self.mind.selectAction(state)
            next_state, reward, done, __ = self.env.step(action)
            self.env.render(render)
            if render:
                time.sleep(0.1)
            step                         = EpisodeStep(state, action, reward,next_state, done)
            episodeReward += reward
            episodeData.append(step)
            state = next_state
        if self.rewardMapper != None:
            self.rewardMapper(episodeData)
        commons.totalEpisodes      += 1
        self.agentEpisode   += 1
        self.discountRewards(episodeData)
        self.mind.pushData(episodeData)
        self.epsilon        *= self.epsilon_decay
        self.epsilon         = 0.01 if self.epsilon < 0.01 else self.epsilon
        return episodeData
        
    
  