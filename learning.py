import mountaincar
from Tilecoder import numTilings, numTiles, tilecode
from pylab import *  # includes numpy
import numpy as np

numRuns = 1
n = numTiles * 3
testnum = 1000

avgreturn = [0] * testnum
avgstep = [0] * testnum

def learn(alpha=.1/numTilings, epsilon=0, numEpisodes=1000):
    theta1 = -0.001*rand(n)
    theta2 = -0.001*rand(n)
    returnSum = 0.0
    for episodeNum in range(numEpisodes):
        G = 0
        # your code goes here (20-30 lines, depending on modularity)
        step = 0
        state = mountaincar.init()
        while (state != None):
            step += 1
            tileIndices = [-1] * numTilings
            tilecode(state[0], state[1], tileIndices)

            rndProb = np.random.random()
            if rndProb < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                theta = theta1 + theta2
                Q = np.array([getQ(theta,tileIndices,0), getQ(theta,tileIndices,1), getQ(theta,tileIndices,2)])
                action = np.argmax(Q)

            reward, statePrime = mountaincar.sample(state, action)
            G += reward

            if statePrime != None:
                tileIndicesPrime = [-1] * numTilings
                tilecode(statePrime[0], statePrime[1], tileIndicesPrime)
                updProb = np.random.choice([1, 2])
                if updProb == 1:
                    # update Q1 (theta1)
                    QPrime = np.array([getQ(theta1, tileIndicesPrime, 0), getQ(theta1, tileIndicesPrime, 1),
                                       getQ(theta1, tileIndicesPrime, 2)])
                    actionPrime = np.argmax(QPrime)
                    valuePrime = getQ(theta2, tileIndicesPrime, actionPrime)
                    value = getQ(theta1, tileIndices, action)
                    for i in tileIndices:
                        theta1[i + action * numTiles] += + alpha * (reward + valuePrime - value)
                else:
                    # update Q2 (theta2)
                    QPrime = np.array([getQ(theta2, tileIndicesPrime, 0), getQ(theta2, tileIndicesPrime, 1),
                                       getQ(theta2, tileIndicesPrime, 2)])
                    actionPrime = np.argmax(QPrime)
                    valuePrime = getQ(theta1, tileIndicesPrime, actionPrime)
                    value = getQ(theta2, tileIndices, action)
                    for i in tileIndices:
                        theta2[i + action * numTiles] += + alpha * (reward + valuePrime - value)
                state = statePrime
            else:
                updProb = np.random.choice([1, 2])
                if updProb == 1:
                    value = getQ(theta1, tileIndices, action)
                    for i in tileIndices:
                        theta1[i + action * numTiles] += + alpha * (reward - value)
                    break
                else:
                    value = getQ(theta2, tileIndices, action)
                    for i in tileIndices:
                        theta2[i + action * numTiles] += + alpha * (reward - value)
                    break
        print("Episode: ", episodeNum, "Steps:", step, "Return: ", G)
        returnSum = returnSum + G
        avgstep[episodeNum] += step
        avgreturn[episodeNum] += G
    print("Average return:", returnSum / numEpisodes)
    return returnSum, theta1, theta2


def getQ(theta, tileIndices, action):
    q = 0
    for i in tileIndices:
        q += theta[i + action * numTiles]
    return q

def Qs(tileIndices, theta1, theta2):
    theta = (theta1 + theta2) * 1/2
    q0 = getQ(theta, tileIndices, 0)
    q1 = getQ(theta, tileIndices, 1)
    q2 = getQ(theta, tileIndices, 2)
    return np.array([q0,q1,q2])

def writeF(theta1, theta2):
    fout = open('value', 'w')
    steps = 50
    tileIndices = [-1] * numTilings
    for i in range(steps):
        for j in range(steps):
            F = tilecode(-1.2 + i * 1.7 / steps, -0.07 + j * 0.14 / steps, tileIndices)
            height = -max(Qs(F, theta1, theta2))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()


def writeS():
    fout = open('step', 'w')
    for i in range(testnum):
        fout.write(repr(avgstep[i]))
        fout.write('\n')
    fout.close()


def writeR():
    fout = open('return', 'w')
    for i in range(testnum):
        fout.write(repr(avgreturn[i]))
        fout.write('\n')
    fout.close()

if __name__ == '__main__':
    runSum = 0.0
    for run in range(numRuns):
        returnSum, theta1, theta2 = learn()
        runSum += returnSum
    writeF(theta1, theta2)
    writeR()
    writeS()
    print("Overall performance: Average sum of return per run:", runSum/numRuns)
