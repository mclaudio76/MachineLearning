import commons
from   a3c import A3C

def train():

    def trainEndedEvaluator(episodeList):
        if len(episodeList) < 10:
            return False
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
                if position >= 0.5:
                    solved += 1
        avg = sumRew / len(actualList)
        avgChanges = changes / len(actualList)
        avgSpeed = velocity / (len(actualList) * 200)
        print("Avg Changes", avgChanges, " Average reward  after ", commons.totalEpisodes, " episodes is ", avg, "Solved = ", solved, " best range (",minP, ", ", maxP,") , avg speed ",avgSpeed)
        if solved / 10.0 > 0.9:
            return True
        
    
    def remapReward(episodeData):
        x0          = -0.5 
        totalReward  = 0
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
    a3c.train(stopFunction=trainEndedEvaluator,remapRewardFunction=remapReward, epsilon=1.0)

def playMountain():
    a3c = A3C('MountainCar-v0')
    a3c.play()

train()

#playMountain()

