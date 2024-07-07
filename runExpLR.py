from BaseTrainers import BaseLR
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
from sklearn.linear_model import LinearRegression
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

    rngData = np.random.default_rng(seed = 888)
    # rngProb = np.random.default_rng(seed = 999)

    # meanX = rngProb.uniform(1.1, 1.9, 10)
    d = 50
    meanX = np.linspace(1, 10, num = d)
    # beta = rngProb.uniform(1, 20, 10)
    beta = np.linspace(-10, 10, num = d)
    noiseShape = 2.1

    def sampler(n: int) -> NDArray:
        XSample = rngData.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        noise: NDArray = stats.lomax.rvs(noiseShape, size = n, random_state = rngData) \
            - stats.lomax.rvs(noiseShape, size = n, random_state = rngData)
        YSample = np.dot(XSample, np.reshape(beta, (-1,1))) + noise.reshape(-1, 1)
        return np.hstack((YSample, XSample))

    def evaluator(trainingResult: LinearRegression) -> float:
        error = trainingResult.coef_ - beta
        XVars = (2 * meanX)**2 / 12
        return np.dot(meanX, error) ** 2 + np.sum(error**2 * XVars)

    lr = BaseLR()

    sampleSizeList = [2**i for i in range(10, 17)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 200
    
    pipeline(resultDir,
             lr, 
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