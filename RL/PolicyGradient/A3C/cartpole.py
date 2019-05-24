import commons
from   a3c import A3C

def train():
    def trainEndedEvaluator(episodeList):
        if len(episodeList) < 100:
            return False
        actualList = episodeList[-100:]
        solved = 0
        totalRew = 0
        for episode in actualList:
            solved = solved+1 if len(episode[0]) >= 450 else solved
            totalRew += len(episode[0])
        avg = totalRew / len(actualList)
        print("Average reward  after ", commons.totalEpisodes, " episodes is ", avg, "Solved = ", solved)
        return solved / len(actualList) > 0.9
            
    a3c = A3C('CartPole-v1',numThreads=10)
    a3c.train(stopFunction=trainEndedEvaluator, epsilon=1.0)

def play():
    a3c = A3C('CartPole-v1')
    __ , rew = a3c.play()
    print(rew)

play()
#train()
