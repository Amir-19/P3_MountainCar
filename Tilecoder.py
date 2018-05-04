import numpy as np
from math import *

numTilings = 4
numTiles = 9 * 9 * numTilings

def tilecode(in1, in2, tileIndices):
    # write your tilecoder here (5 lines or so)
    offsetPos = ((1/numTilings) * (1.7/8)) * np.arange(numTilings)
    offsetVel = ((1/numTilings) * (0.14/8)) * np.arange(numTilings)
    x = np.floor(((in1+1.2) + offsetPos)/(1.7/8)).astype(int)
    y = (np.floor(((in2+0.07) + offsetVel)/(0.14/8))*9).astype(int)
    tileIndices[:] = np.floor(81 * np.arange(numTilings) + x + y).astype(int)
    return tileIndices



def printTileCoderIndices(in1, in2):
    tileIndices = [-1] * numTilings
    tilecode(in1, in2,tileIndices)
    print('Tile indices for input (', in1, ',', in2,') are : ', tileIndices)


def testTileCoder():
    printTileCoderIndices(-0.3, -0.04)
    printTileCoderIndices(0.2, +0.01)
    printTileCoderIndices(-0.1, +0)
    printTileCoderIndices(0.4, -0.06)




