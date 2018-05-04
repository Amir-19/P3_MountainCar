from pylab import *

def plotSteps():

    x1 = np.arange(200)
    y1 = loadtxt("step-1-4")
    y1 = y1/50
    plt.plot(x1, y1,label='alpha = 0.1/4')

    plt.xlabel('Num of Episodes')
    plt.ylabel('Average Steps')
    plt.title('Average Number of Steps double Q-Learning in 50 Runs')
    plt.savefig("steps.pdf")
    plt.show()

def plotReturns():
    x = np.arange(200)
    y = loadtxt("return-1-4")
    y = y/50
    plt.plot(x, y)

    plt.xlabel('Num of Episodes')
    plt.ylabel('Average Return')
    plt.title('Average Return Steps double Q-Learning in 50 Runs')
    plt.savefig("returns.pdf")
    plt.show()

plotSteps()