from ActorCritic import A2C
from keras.optimizers import Adam, RMSprop

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


    a2c = A2C('CartPole-v1', actor_optimizer=Adam(lr=0.0001), critic_optimizer=Adam(lr=0.002), epsilon=0.1, epsilon_decay=0.9)
    a2c.train(logger=logInfo, trainEndedEvaluator= trainEndedEvaluator)

main()