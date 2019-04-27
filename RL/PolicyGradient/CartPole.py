from ActorCritic import A2C

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