from DQAgent import DQAgent, EpisodeStep

DDQEnabled          = True
TRAIN_STEP          = 25
EPISODE_NUMBER      = 10000
SWAP_WEIGHTS_STEP   = 50
EVAL_EPISODES       = 100

def sumReward(result, numEpisodes):
    if len(result) < numEpisodes:
        return 0.0
    lastNRuns = result[-numEpisodes:]
    cumulativeReward = 0.0
    for run  in lastNRuns:
        gameReward = 0.0
        for epsStep in run:
            gameReward += epsStep.reward
        cumulativeReward += gameReward
    return cumulativeReward


def train_cartPole():
   
    def averageOnLast(result, numEpisodes):
        return sumReward(result, numEpisodes) / numEpisodes

    def stopFunction(results):
        return averageOnLast(results,EVAL_EPISODES) >= 450.0

    def printStat(results):
        avg = averageOnLast(results,TRAIN_STEP)
        print(" Average score of last ", TRAIN_STEP, " episodes ",avg) 

    def custReward(episodeData):
        for epsStep in episodeData:
            epsStep.reward = epsStep.reward if not epsStep.done else 0

    agent = DQAgent('CartPole-v1',memoryBufferSize=100000, epsilon_decay=0.95, train=True, DDQEnabled=DDQEnabled)
    
    agent.train(stopFunction=stopFunction, trainStep=TRAIN_STEP, swapStep=50, numEpisodes=EPISODE_NUMBER, customize_reward_function=custReward, progress_log_funct=printStat)


def play_cartPole():
    agent = DQAgent('CartPole-v1',memoryBufferSize=100000, epsilon_decay=0.7, train=False,DDQEnabled=DDQEnabled)
    data = agent.playSingleEpisode(render=True)
    d    = list()
    d.append(data)
    rew = sumReward ( d, 1 )
    print(" Reward = ",rew)