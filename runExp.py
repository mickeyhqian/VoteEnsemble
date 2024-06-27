from BaseTrainers import BaseLR
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
from sklearn.linear_model import LinearRegression
import os
from uuid import uuid4



if __name__ == "__main__":
    if os.name == "posix":
        set_start_method("fork")

    rngData = np.random.default_rng(seed = 888)
    rngProb = np.random.default_rng(seed = 999)

    meanX = rngProb.uniform(1.1, 1.9, 10)
    beta = rngProb.uniform(1, 20, 10)
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

    sampleSizeList = [2**i for i in range(10, 15)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 10
    
    pipeline("LR_SAA", 
             str(uuid4()), 
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
             numParallelEval = 12)