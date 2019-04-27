import random
import numpy as np
import gym
import keras.backend as K
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, SGD
import math


class EpisodeStep:
    def __init__(self, state, action, reward, next_state, done):
        self.state      = state
        self.action     = action
        self.reward     = reward
        self.next_state = next_state
        self.done       = done


class Actor():

    def __init__(self, inp_dim, out_dim, base_network, optimizer):
        
        self.inp_dim        = inp_dim
        self.out_dim        = out_dim
      

        hiddenLayer1        = Dense(128, activation='relu') (base_network.output)
        hiddenLayer2        = Dense(64,  activation='relu') (hiddenLayer1)
        agentOutput         = Dense(self.out_dim, activation='softmax') (hiddenLayer2)

        self.__model        = Model(base_network.input, agentOutput)
        self.action_pl      = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl  = K.placeholder(shape=(None,))
   
        weighted_actions    = K.sum(self.action_pl * self.model().output, axis=1)
        eligibility         = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy             = K.sum(self.model().output * K.log(self.model().output + 1e-10), axis=1)
        loss                = 0.001 * entropy - K.sum(eligibility)
        updates             = optimizer.get_updates(self.model().trainable_weights, [], loss)
        self.__opt          = K.function([self.model().input, self.action_pl, self.advantages_pl], [], updates=updates)
        

    def optimize(self, inputs):
       self.__opt(inputs)

    def model(self):
        return self.__model

    def save(self, path):
        self.model().save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model().load_weights(path)


class Critic():

    def __init__(self, inp_dim, out_dim, base_network, optimizer):
        self.inp_dim        = inp_dim
        self.out_dim        = out_dim


        hiddenLayer1        = Dense(128, activation='relu') (base_network.output)
        agentOutput         = Dense(1,   activation='linear') (hiddenLayer1)



        self.__model        = Model(base_network.input, agentOutput) 
        self.discounted_r   = K.placeholder(shape=(None,))
        critic_loss         = K.mean(K.square(self.discounted_r - self.model().output))
        updates             = optimizer.get_updates(self.model().trainable_weights, [], critic_loss)
        self.__optimizer    = K.function([self.model().input, self.discounted_r], [], updates=updates)

    def optimize(self, inputs):
        self.__optimizer(inputs)

    def model(self):
        return self.__model

    def save(self, path):
        self.model().save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model().load_weights(path)


class A2C:
    def __init__(self, enviromentName:str, training=True, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        self.enviromentName    = enviromentName
        self.env               = gym.make(enviromentName) # creates enviroment using symbolic name
        self.env.reset()         # reset enviroments
        self.env_dim           = self.env.observation_space.shape[0]
        self.act_dim           = self.env.action_space.n 
        
        # Environment and A2C parameters
        
        self.epsilon        = 0.0
        self.epsilon_decay  = 0.9
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.__base_network()
        self.actor  = Actor (self.env_dim,  self.act_dim, self.shared, Adam(lr=0.0001))
        self.critic = Critic(self.env_dim,  self.act_dim, self.shared, Adam(lr=0.0001))
        if not training:
            self.load_weights()


    def __base_network(self):
        inp       = Input(shape=(self.env_dim,))
        hidden1   = Dense(64, activation='relu')(inp)
        hidden2   = Dense(128, activation='relu')(hidden1)
        return Model(inp, hidden2)

    def selectAction(self, state):
        if np.random.random() < self.epsilon:
           return self.env.action_space.sample()
        s = self.__reshapeStateWhenNeeded(state)
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.model().predict(s).ravel())[0]

    def __reshapeStateWhenNeeded(self, state):
        if len(state.shape) < 3: 
            x = np.expand_dims(state, axis=0)
            return x
        return state

    def discount(self, r):
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

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


    def __train_models(self, episodeData):
        states  = []
        actions = []
        rewards = []
        for eps in episodeData :
            states.append(eps.state)
            actions.append(to_categorical(eps.action, self.act_dim))
            rewards.append(eps.reward)
        discounted_rewards  = self.discount(rewards)
        np_state            = np.array(states)
        state_values        = self.critic.model().predict(np_state)
        advantages          = discounted_rewards - np.reshape(state_values, len(state_values))
        self.actor.optimize([states, actions, advantages])
        self.critic.optimize([states, discounted_rewards])


    

    def train(self, trainEndedEvaluator, numEpisodes = 10000, trainStep=100, remapRewardFunction=None, logger=None):
        results = []
        # Main Loop
        for e in range(numEpisodes):
            if e % trainStep == 0 and e > 0:
               self.epsilon *= self.epsilon_decay
               self.epsilon  = 0.0 if self.epsilon < 0.001 else self.epsilon
            # Reset episode
            runData = self.playSingleEpisode()
            if logger != None:
                logger(e, runData)
            if remapRewardFunction != None:
                for eps in runData:
                    remapRewardFunction(eps)
            self.__train_models(runData)
            results.append(runData)
            if trainEndedEvaluator(results):
                self.save_weights()
                print ("Training ended")
                return results
        return results

    def save_weights(self):
        self.actor.save(self.enviromentName+"-actor.mod")
        self.critic.save(self.enviromentName+"-critic.mod")

    def load_weights(self):
        self.actor.load_weights(self.enviromentName+"-actor.mod")
        self.critic.load_weights(self.enviromentName+"-critic.mod")


def main():
    
    def trainEndedEvaluator(episodeList):
        actualList = episodeList[-100:]
        sumRew = 0
        for episode in actualList:
            for epsState in episode:
                sumRew += epsState.reward
        return sumRew / 100 > 450
    
    def logInfo(episode, runData):
        if episode % 50 == 0:
            totalReward = 0
            for eps in runData:
                totalReward += eps.reward
            print(" Episode ", episode, " reward ", totalReward)


    a2c = A2C('CartPole-v1')
    a2c.train(logger=logInfo, trainEndedEvaluator= trainEndedEvaluator)

main()