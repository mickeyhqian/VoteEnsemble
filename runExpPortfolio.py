from BaseTrainers import BasePortfolio
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
from uuid import uuid4
import sys
import os
import logging
logger = logging.getLogger(name = "Bagging")



if __name__ == "__main__":
    set_start_method("spawn")

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
    # meanX = rngProb.uniform(1.95, 1.98, d)
    # meanX = np.linspace(1.95, 1.99, num = d)
    genMatrix = []
    for i in range(0, hiddenD, hiddenD // d):
        vec = np.zeros(hiddenD)
        vec[i] = 1/2
        genMatrix.append(np.full(hiddenD, 1/2 / hiddenD) + vec)
    genMatrix = np.asarray(genMatrix).T
    varMatrix = np.dot(genMatrix.T, np.dot(np.diag(hiddenVars), genMatrix))
    meanVector = np.dot(hiddenMeans, genMatrix)
    b = 1.3

    # def sampler(n: int) -> NDArray:
    #     return stats.lomax.rvs(meanX / (meanX - 1), size = (n, len(meanX)), random_state = rngData) + 1

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        hidden = stats.lomax.rvs(hiddenShapes, size = (n, hiddenD), random_state = rng) * hiddenMultipliers.reshape(1, -1)
        return np.dot(hidden, genMatrix)

    # def evaluator(trainingResult: NDArray) -> float:
    #     shapes = meanX / (meanX - 1)
    #     variance = shapes / ((shapes-1)**2 * (shapes-2))
    #     return np.sum(variance * trainingResult**2)

    def evaluator(trainingResult: NDArray, repIdx: int) -> float:
        return np.dot(trainingResult, np.dot(varMatrix, trainingResult))

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
             numParallelTrain = 1, 
             numParallelEval = 1)