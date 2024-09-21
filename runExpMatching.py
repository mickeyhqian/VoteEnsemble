from BaseLearners import BaseMatching
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from uuid import uuid4
import sys
import os
import logging
logger = logging.getLogger(name = "VE")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        resultDir = sys.argv[1]
    else:
        resultDir = os.path.join(os.path.dirname(__file__), str(uuid4()))

    os.makedirs(resultDir, exist_ok = True)
    logger.setLevel(logging.DEBUG)
    logHandler = logging.FileHandler(os.path.join(resultDir, "exp.log"))
    formatter = logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(message)s")
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    
    N = 5
    randomN = 3
    epsilon = 0
    # epsilon = 0.25
    w = {
        (0, 0): 1.8865364093012886,
        (0, 1): 2.398555301120877,
        (0, 2): 1.8306427790927857,
        (0, 3): 1.7492991192565832,
        (0, 4): 2.3237863958023786,
        (1, 0): 2.137122473241966,
        (1, 1): 2.2498292819020653,
        (1, 2): 1.8709355265561154,
        (1, 3): 1.7336844551004142,
        (1, 4): 1.8512613494646823,
        (2, 0): 1.873453484656252,
        (2, 1): 2.32957861213437,
        (2, 2): 2.2815754847013983,
        (2, 3): 2.1955418952557166,
        (2, 4): 1.9664292773529026,
        (3, 0): 1.6368204540890734,
        (3, 1): 2.1733533180049087,
        (3, 2): 2.29142702055407,
        (3, 3): 1.64693564175383,
        (3, 4): 2.2760005110017376,
        (4, 0): 2.390491551306702,
        (4, 1): 2.340076629674212,
        (4, 2): 1.8125406416083787,
        (4, 3): 1.9427529395181724,
        (4, 4): 1.6101934594984615,
    }

    paretoParams = []
    for i in range(randomN):
        for j in range(randomN):
            paretoParams.append(w[(i,j)] / (w[(i,j)] - 1))

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        arraysList = []
        for param in paretoParams:
            newArray = stats.lomax.rvs(param, size = n, random_state = rng) + 1
            arraysList.append(newArray)
        return np.asarray(arraysList).T

    def evaluator(learningResult: NDArray, repIdx: int) -> float:
        idx = 0
        output = 0.0
        for i in range(N):
            for j in range(N):
                output += w[(i, j)] * learningResult[idx]
                idx += 1
        return -output

    matching = BaseMatching(w, N, randomN, epsilon)

    sampleSizeList = [2**i for i in range(10, 16)]
    kList = [(10, 0.005)]
    BList = [200]
    k12List = [((10, 0.005), (10, 0.005))]
    B12List = [(20, 200)]
    numReplicates = 200
    
    pipeline(resultDir,
             matching, 
             sampler, 
             evaluator, 
             sampleSizeList, 
             kList, 
             BList, 
             k12List, 
             B12List, 
             numReplicates, 
             numParallelLearn = 1, 
             numParallelEval = 1)