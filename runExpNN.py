from BaseTrainers import BaseNN, RegressionNN
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
from uuid import uuid4



if __name__ == "__main__":
    set_start_method("spawn")

    rngData = np.random.default_rng(seed = 888)
    # rngProb = np.random.default_rng(seed = 999)
    rngEval = np.random.default_rng(seed = 777)

    d = 10
    # meanX = rngProb.uniform(1.1, 1.9, d)
    meanX = np.linspace(1, 100, num = d)
    noiseShape = 2.1

    def trueMapping(x: NDArray) -> float:
        return np.mean(np.log(x + 1))

    def sampler(n: int) -> NDArray:
        XSample = rngData.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        noise: NDArray = stats.lomax.rvs(noiseShape, size = n, random_state = rngData) \
            - stats.lomax.rvs(noiseShape, size = n, random_state = rngData)
        YSample = np.asarray([[trueMapping(x)] for x in XSample]) + noise.reshape(-1, 1)
        return np.hstack((YSample, XSample))
    
    def evalSampler(n: int) -> NDArray:
        XSample = rngEval.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        YSample = np.asarray([[trueMapping(x)] for x in XSample])
        return np.hstack((YSample, XSample))
    
    baseNN = BaseNN([50, 50], useGPU = False)

    evalSample = evalSampler(1000000)

    def evaluator(trainingResult: RegressionNN) -> float:
        return baseNN.objective(trainingResult, evalSample)


    sampleSizeList = [2**i for i in range(14, 20)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 100
    
    pipeline(f"NN_d{d}_SAA", 
             str(uuid4()), 
             baseNN, 
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