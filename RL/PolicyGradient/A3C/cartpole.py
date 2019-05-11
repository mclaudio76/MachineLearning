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
            episodeRew = 0
            solved = solved+1 if len(actualList) >= 450 else solved
            for item in episode:
               episodeRew += item.reward
            totalRew += episodeRew
        avg = totalRew / len(actualList)
        print("Average reward  after ", commons.totalEpisodes, " episodes is ", avg, "Solved = ", solved)
        return avg >= 450
            
    a3c = A3C('CartPole-v1',numThreads=10)
    a3c.train(stopFunction=trainEndedEvaluator, epsilon=1.0)

def play():
    a3c = A3C('CartPole-v1')
    a3c.play()

play()
#train()
