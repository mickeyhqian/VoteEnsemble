from BaseLearners import BaseLR
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from uuid import uuid4
import sys
import os
import pandas as pd
import logging
logger = logging.getLogger(name = "VE")



if __name__ == "__main__":
    dataFile = sys.argv[1]
    
    if len(sys.argv) > 1:
        resultDir = sys.argv[2]
    else:
        resultDir = os.path.join(os.path.dirname(__file__), str(uuid4()))

    os.makedirs(resultDir, exist_ok = True)
    logger.setLevel(logging.DEBUG)
    logHandler = logging.FileHandler(os.path.join(resultDir, "exp.log"))
    formatter = logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(message)s")
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    rngData = np.random.default_rng(seed = 888)

    data = pd.read_csv(dataFile).to_numpy()
    data = data[rngData.permutation(len(data))]
    
    trainSize = len(data) // 2
    testSize = len(data) - trainSize
    
    numReplicates = 100
    seedList = rngData.choice(1000000, size = numReplicates, replace = False)

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        rng = np.random.default_rng(seed = seedList[repIdx])
        select = np.full(len(data), False)
        select[rng.choice(len(data), size = trainSize, replace = False)] = True
        return data[select]
    
    baseLR = BaseLR()

    def evaluator(learningResult: LinearRegression, repIdx: int) -> float:
        rng = np.random.default_rng(seed = seedList[repIdx])
        select = np.full(len(data), True)
        select[rng.choice(len(data), size = trainSize, replace = False)] = False
        return baseLR.objective(learningResult, data[select]).mean()

    lr = BaseLR()

    sampleSizeList = [trainSize]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    
    pipeline(resultDir,
             lr, 
             sampler, 
             evaluator, 
             None,
             None,
             sampleSizeList, 
             kList, 
             BList, 
             k12List, 
             B12List, 
             numReplicates, 
             numParallelLearn = 1, 
             numParallelEval = 1,
             dumpSubsampleResults = True,
             runConventionalBagging = False)