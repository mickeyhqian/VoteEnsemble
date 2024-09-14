from BaseLearners import BaseNN, RegressionNN
from ExpPipeline import pipeline
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from multiprocessing import set_start_method
from uuid import uuid4
import sys
import os
import torch
import logging
logger = logging.getLogger(name = "VE")



if __name__ == "__main__":
    set_start_method("spawn")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    dataFile = sys.argv[1]

    if len(sys.argv) > 2:
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

    sampleSizeList = [trainSize]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 100
    seedList = rngData.choice(1000000, size = numReplicates, replace = False)

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        rng = np.random.default_rng(seed = seedList[repIdx])
        select = np.full(len(data), False)
        select[rng.choice(len(data), size = trainSize, replace = False)] = True
        return data[select]
    
    # baseNN = BaseNN([50, 300, 500, 800, 800, 500, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 300, 500, 500, 300, 50], learningRate = 0.001, useGPU = False)
    baseNN = BaseNN([100, 300, 300, 100], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 50], learningRate = 0.001, useGPU = False)

    evalDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluator(learningResult: RegressionNN, repIdx: int) -> float:
        rng = np.random.default_rng(seed = seedList[repIdx])
        select = np.full(len(data), True)
        select[rng.choice(len(data), size = trainSize, replace = False)] = False
        return baseNN.objective(learningResult, data[select], device = evalDevice).mean()

    
    pipeline(resultDir,
             baseNN, 
             sampler, 
             evaluator, 
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