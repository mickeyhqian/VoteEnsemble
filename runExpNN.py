from BaseLearners import BaseNN, RegressionNN
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
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

    rngEval = np.random.default_rng(seed = 777)

    d = 50
    meanX = np.linspace(1, 100, num = d)
    noiseShape = 2.1

    def trueMapping(x: NDArray) -> float:
        return np.mean(np.log(x + 1))

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        XSample = rng.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        # noise: NDArray = stats.lomax.rvs(noiseShape, size = n, random_state = rng) \
        #     - stats.lomax.rvs(noiseShape, size = n, random_state = rng)
        noise: NDArray = rng.normal(size = n)
        YSample = np.asarray([[trueMapping(x)] for x in XSample]) + noise.reshape(-1, 1)
        return np.hstack((YSample, XSample))
    
    def evalSampler(n: int) -> NDArray:
        XSample = rngEval.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        YSample = np.asarray([[trueMapping(x)] for x in XSample])
        return np.hstack((YSample, XSample))
    
    baseNN = BaseNN([50, 300, 500, 800, 800, 500, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 300, 500, 500, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 300, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 50], learningRate = 0.001, useGPU = False)

    evalSample = evalSampler(1000000)
    evalDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluator(learningResult: RegressionNN, repIndex: int) -> float:
        return baseNN.objective(learningResult, evalSample, device = evalDevice)


    sampleSizeList = [2**i for i in range(10, 17)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 100

    
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