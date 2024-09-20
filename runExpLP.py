from BaseLearners import BaseLP
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
    
    N = 8
    A = np.ones((N, N))
    w = {
        (1, 2): 3,
        (1, 3): 3,
        (1, 4): 3,
        (1, 5): 3.5,
        (1, 6): 3.5,
        (1, 7): 3.5,
        (1, 8): 3.5,
        (2, 3): 3,
        (2, 4): 3,
        (2, 5): 3.5,
        (2, 6): 3.5,
        (2, 7): 3.5,
        (2, 8): 3.5,
        (3, 4): 3,
        (3, 5): 3.5,
        (3, 6): 3.5,
        (3, 7): 3.5,
        (3, 8): 3.5,
        (4, 5): 3.5,
        (4, 6): 3.5,
        (4, 7): 3.5,
        (4, 8): 3.5,
        (5, 6): 3.5,
        (5, 7): 3.5,
        (5, 8): 3.5,
        (6, 7): 3.5,
        (6, 8): 3.5,
        (7, 8): 3.5,
    }

    paretoParams = []
    for i in range(1, N):
        for j in range(i + 1, N + 1):
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
            for j in range(i + 1, N):
                output += w[(i + 1, j + 1)] * learningResult[idx]
                idx += 1
        return -output

    lp = BaseLP(A)

    sampleSizeList = [2**i for i in range(8, 15)]
    kList = [(10, 0.005)]
    BList = [200]
    k12List = [((10, 0.005), (10, 0.005))]
    B12List = [(20, 200)]
    numReplicates = 200
    
    pipeline(resultDir,
             lp, 
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