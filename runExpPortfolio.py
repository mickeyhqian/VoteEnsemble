from BaseLearners import BasePortfolio
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

    d = 10
    hiddenD = 100
    hiddenShapes = np.full(hiddenD, 2.01)
    hiddenMeans = np.linspace(0.5, 2, num = hiddenD)
    hiddenMultipliers = (hiddenShapes - 1) * hiddenMeans
    hiddenVars = hiddenMultipliers**2 * hiddenShapes / ((hiddenShapes - 1)**2 * (hiddenShapes - 2))
    genMatrix = []
    for i in range(0, hiddenD, hiddenD // d):
        vec = np.zeros(hiddenD)
        vec[i] = 1/2
        genMatrix.append(np.full(hiddenD, 1/2 / hiddenD) + vec)
    genMatrix = np.asarray(genMatrix).T
    varMatrix = np.dot(genMatrix.T, np.dot(np.diag(hiddenVars), genMatrix))
    meanVector = np.dot(hiddenMeans, genMatrix)
    b = 1.3

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        hidden = stats.lomax.rvs(hiddenShapes, size = (n, hiddenD), random_state = rng) * hiddenMultipliers.reshape(1, -1)
        return np.dot(hidden, genMatrix)

    def evaluator(learningResult: NDArray, repIdx: int) -> float:
        return np.dot(learningResult, np.dot(varMatrix, learningResult))

    portfolio = BasePortfolio(meanVector, b)

    sampleSizeList = [2**i for i in range(10, 17)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 200
    
    pipeline(resultDir,
             portfolio, 
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