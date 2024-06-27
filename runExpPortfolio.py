from BaseTrainers import BasePortfolio
from ExpPipeline import pipeline
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from scipy import stats
from uuid import uuid4



if __name__ == "__main__":
    set_start_method("spawn")

    rngData = np.random.default_rng(seed = 888)
    rngProb = np.random.default_rng(seed = 999)

    meanX = rngProb.uniform(1.95, 1.98, 10)
    b = 1.96

    def sampler(n: int) -> NDArray:
        return stats.lomax.rvs(meanX / (meanX - 1), size = (n, len(meanX)), random_state = rngData) + 1

    def evaluator(trainingResult: NDArray) -> float:
        shapes = meanX / (meanX - 1)
        variance = shapes / ((shapes-1)**2 * (shapes-2))
        return np.sum(variance * trainingResult**2)

    portfolio = BasePortfolio(meanX, b)

    sampleSizeList = [2**i for i in range(10, 15)]
    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 10
    
    pipeline("portfolio_SAA", 
             str(uuid4()), 
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