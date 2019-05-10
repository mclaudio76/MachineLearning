import gym
import math,threading,time
from collections      import deque   

class Enviroment:

    def __init__(self, envName, seed=42):
        self.envName = envName
        self.env      = gym.make(envName)
        self.env.reset()    
        self.env.seed(seed)   
        self.env_dim           = self.env.observation_space.shape[0]
        self.act_dim           = self.env.action_space.n 

    def reset(self):
        return self.env.reset()

    def envDim(self):
        return self.env_dim
    
    def actDim(self):
        return self.act_dim

    def step(self, action):
        return self.env.step(action)

    def randomAction(self):
        return self.env.action_space.sample()
    
    def render(self, display=False):
        if display:
            self.env.render()

class History:
    def __init__(self):
        self.lock_queue        = threading.Lock()
        self.results           = deque(maxlen=1000000)

    def registerResult(self,episodeData):
        with self.lock_queue:
           self.results.append(episodeData)


class EpisodeStep:
    def __init__(self, state, action, reward, next_state, done):
        self.orig_state      = state
        self.orig_next_state = next_state
        self.state      = self.remapState(state)
        self.action     = action
        self.reward     = reward
        self.next_state = self.remapState(next_state)
        self.done       = done

    def remapState(self, state):
        return state 
