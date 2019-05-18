import gym
import random
import sys
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models     import Sequential
from keras.layers     import Dense,Conv2D,Lambda,Input,Convolution2D, Flatten
from keras.optimizers import Adam, SGD,RMSprop
from keras.models     import load_model, Model
from collections      import deque    
import time

'''
Mauri Claudio, 28.04.2019

Playing break-out using Double Deep Q Learning.

'''

FRAME_WIDTH  = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4   # Number of most recent frames to produce the input to the network

class EpisodeStep:
    def __init__(self, state, action, reward, next_state, done):
        self.state      = state
        self.action     = action
        self.reward     = reward
        self.next_state = next_state
        self.done       = done

class DQBreakOut:
    
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
       
        
        self.memory             = deque(maxlen=memoryBufferSize)

        self.convNet            = self.__buildConvNet()

        self.agent_model        = self.__buildModel(self.convNet, self.action_size)
        self.qfunction_model    = self.__buildModel(self.convNet, self.action_size)
        

        self.trainingMode   = train

        if not self.trainingMode:
            self.loadModels()
        self.agent_model.summary()
  
    def __buildModel(self, convNet, output_size):
        hLayer1 = Dense(128, activation='relu' , kernel_initializer='he_uniform')(convNet.output)
        #hLayer2 = Dense(128, activation='relu' , kernel_initializer='he_uniform')(hLayer1)
        #hLayer3 = Dense(64, activation='relu' , kernel_initializer='he_uniform')(hLayer2)
        output  = Dense(output_size, activation='linear')(hLayer1)
        model   = Model(inputs=convNet.input, outputs=output)
        optimizer = Adam(lr=0.005)
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    def __buildConvNet(self):
         # With the functional API we need to define the inputs.
        frames_input = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH), name='frames')
        normalized   = Lambda(lambda x: x / 255.0)(frames_input)
        conv_1       = Convolution2D( 16, (8, 8) , strides=(4, 4), activation='relu', kernel_initializer='he_uniform')(normalized)
        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2       = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        #hidden      = Dense(256, activation='relu')(conv_flattened)
        #outlayer    = Dense(256)(hidden)
        outlayer     = Dense(256, activation='relu')(conv_flattened)
        model       = Model(inputs=frames_input, outputs=outlayer)
        return model

   


    def computeQValue(self, model, state):
        transformed_state = np.reshape([state], (1, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH))
        rewards  = model.predict(transformed_state)[0]
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

    
    def selectAction(self,state):
        if self.trainingMode and (np.random.random() <= self.epsilon) :
            action =  self.env.action_space.sample()
        else:
            prediction = self.computeQValue(self.agent_model,state)
            action = np.argmax(prediction)
        return action


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
        
        X = np.array([x for x in state_array]).reshape(( len(state_array), FRAME_WIDTH,FRAME_HEIGHT,STATE_LENGTH))
        y = np.array([y for y in Qvalues_array]).reshape(-1, self.action_size)
        self.agent_model.fit(X, y, epochs=1, verbose=0)

      
    def preprocessState(self, state):
        grayscaled_reduced_state         = np.uint8( resize(rgb2gray(state), (FRAME_WIDTH, FRAME_HEIGHT), mode='constant') * 255)
        return grayscaled_reduced_state

    def augmentState(self, processedState, episodeData):
        if len(episodeData) == 0:
            return (processedState,processedState, processedState, processedState)
        else:
            lastEpisode = episodeData[-1]
            return (lastEpisode.next_state[1], lastEpisode.next_state[2], lastEpisode.next_state[3], processedState)

    '''
    Collects and returns the whole episode steps' data.
    '''
    def playSingleEpisode(self, render=False):
        episodeData = list()
        state       = self.env.reset()
        self.env.step(1) #press fire ??
        done        = False
        state       = self.preprocessState(state) 
        history_state    = self.augmentState(state,episodeData)
        lives            = 5
        totalReward      = 0
        while not done:
            action                       = self.selectAction(history_state)
            next_state, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
                time.sleep(0.1)
            next_state                   = self.preprocessState(next_state)
            history_next_state           = self.augmentState(next_state, episodeData)
            step                         = EpisodeStep(history_state, action, reward, history_next_state, done)
            
            if info['ale.lives']  < lives:
                lives =   info['ale.lives']
                step.reward = -10 
            
                done = True
            totalReward += step.reward
            episodeData.append(step)
            history_state = history_next_state
           
        return episodeData, totalReward


    def pushToMemory(self, episodeData):      
        for x in episodeData:
            self.memory.appendleft(x)

  
    def train(self, trainStep = 100, swapStep  = 100, numEpisodes=10000, minibatchSize=1000, customize_reward_function = None, progress_log_funct = None):
        results = list()
        totalGoods = 0
        goodScore  = 0
        for currentEpisode in range(numEpisodes):
            episodeData, totalEpisodeReward = self.playSingleEpisode(render=False)
            if customize_reward_function != None:
                customize_reward_function(episodeData)
            if totalEpisodeReward > -10:
                #print("[Good Episode ", currentEpisode, "/",numEpisodes,"]; epsilon ", self.epsilon,"reward ",totalEpisodeReward, end='\n')
                results.append(episodeData)
                self.pushToMemory(episodeData)        
                totalGoods += 1    
                goodScore  += totalEpisodeReward
            if currentEpisode > 0 and currentEpisode % trainStep == 0 and totalGoods > minibatchSize:
                minibatch = random.sample(self.memory, min(len(self.memory), minibatchSize))
                print("[CurrentEpisode]", currentEpisode," Now Training : total goods ",totalGoods, " avg = ",(goodScore / totalGoods), " epsilon = ",self.epsilon)
                time.sleep(1)
                totalGoods = 0
                goodScore  = 0
                self.batch_train(minibatch)
                if progress_log_funct != None:
                    progress_log_funct(results)
                self.saveModels()
                self.epsilon *= self.epsilon_decay
                self.epsilon  = 0 if self.epsilon < 0.001 else self.epsilon
                self.qfunction_model.set_weights(self.agent_model.get_weights())
                self.memory.clear()




def main():

    def progress_log_funct(results):
        if len(results) % 100 == 0:
            lastN = results[-100:]
            totalScore = 0
            for run in lastN:
                for eps in run:
                    totalScore += eps.reward
            print(" Avg score  on last 100 games", totalScore / 100)

    agent = DQBreakOut(enviromentName='BreakoutDeterministic-v4', memoryBufferSize=150000, DDQEnabled=True, train=True)
    agent.train(trainStep=1000,swapStep=10, numEpisodes=1000000, minibatchSize=250, progress_log_funct=progress_log_funct)
    #agent.playSingleEpisode(render=True)

main()