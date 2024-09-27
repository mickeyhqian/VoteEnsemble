from BaseLearners import BaseNetwork
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing.pool import Pool
from scipy import stats
from uuid import uuid4
import sys
import os
import logging
logger = logging.getLogger(name = "VE")



def evaluateObjective(network: BaseNetwork, learningResult: NDArray, sample: NDArray) -> NDArray:
    return network.objective(learningResult, sample)


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
    
    s, p, c, g = 3, 2, 3, 5
    C = np.array([1.5, 40])
    Q_sp = np.concatenate([np.ones((s, 1, g)), 1e-2 * np.ones((s, 1, g))], axis=1)
    Q_pc = np.concatenate([np.ones((1, c, g)), 1e-2 * np.ones((1, c, g))], axis=0)
    R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
                    [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    M = np.array([1000, 1000])
    H = np.ones((c, g)) * 5
    paramS = np.ones((s, g)) * 5
    paramD = np.ones((c, g)) * 5
    paramD[0, 0] = 2.1

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        sampleS = stats.lomax.rvs(paramS, size = (n, s, g), random_state = rng) + 1000
        sampleD = stats.lomax.rvs(paramD, size = (n, c, g), random_state = rng) + 1
        return np.concatenate((sampleS, sampleD), axis = 1)

    rngEval = np.random.default_rng(seed = 777)
    evalSample = sampler(100000, -1, rngEval)
    
    network = BaseNetwork(C, Q_sp, Q_pc, R, M, H)
    numParallel = 14
    
    def evaluator(learningResult: NDArray, repIdx: int) -> float:
        interval = max(1, len(evalSample) // numParallel + 1)
        with Pool(numParallel) as pool:
            results = pool.starmap(
                evaluateObjective,
                [(network, learningResult, evalSample[i:min(len(evalSample), i + interval)]) for i in range(0, len(evalSample), interval)],
                chunksize = 1
            )
        return np.mean(np.concatenate(results))

    sampleSizeList = [2**i for i in range(7, 12)]
    kList = [(10, 0.005)]
    # kList = [(10, 0.1)]
    BList = [200]
    k12List = [((10, 0.005), (10, 0.005))]
    # k12List = [((10, 0.1), (10, 0.1))]
    B12List = [(20, 200)]
    numReplicates = 200
    
    pipeline(resultDir,
             network, 
             sampler, 
             evaluator, 
             sampleSizeList, 
             kList, 
             BList, 
             k12List, 
             B12List, 
             numReplicates, 
             numParallelLearn = numParallel, 
             numParallelEval = numParallel)