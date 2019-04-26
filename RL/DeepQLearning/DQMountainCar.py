from DQAgent import DQAgent, EpisodeStep

DDQEnabled          = False
TRAIN_STEP          = 50
EPISODE_NUMBER      = 100000
EVAL_EPISODES       = 100
BATCH_SIZE          = 5000

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

def averageOnLast(result, numEpisodes):
    return sumReward(result, numEpisodes) / numEpisodes

def rightMostPosition(episode):
    maxPos    = -1.2
    for epsStep in episode:
        maxPos = max(maxPos, epsStep.next_state[0])
    return maxPos

def stopFunction(results):
    if len(results) < 100:
        return False
    lastNRuns = results[-100:]
    solved    = 0
    for run in lastNRuns:
        solved = solved +1 if rightMostPosition(run) >= 0.5 else solved
    return solved / 100 >= 0.9

def printStat(results):
    avg = averageOnLast(results,TRAIN_STEP)
    maxPos = -1.2
    for episode in results:
        maxPos = max(maxPos, rightMostPosition(episode))
    print(" Average score of last ", TRAIN_STEP, " episodes ",avg," max pos ", maxPos) 

def custReward(episodeData):
    for epsStep in episodeData:
        if epsStep.next_state[0] > -0.2:
            epsStep.reward = 1



agent = DQAgent('MountainCar-v0',memoryBufferSize=100000, epsilon_decay=0.95, train=True, DDQEnabled=False)
agent.train(stopFunction=stopFunction, trainStep=TRAIN_STEP, numEpisodes=EPISODE_NUMBER, minibatchSize=BATCH_SIZE, customize_reward_function=custReward, progress_log_funct=printStat)


