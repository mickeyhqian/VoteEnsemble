from BaseLearners import BaseNetwork
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
    
    s, p, c, g = 3, 2, 3, 5
    C = np.array([0.84902017, 0.73141691])
    Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
                        [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
                        [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
                        [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
                        [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
                        [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
                        [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
                        [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
                        [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
                        [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
                        [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
                    [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    M = np.array([0.96766421, 0.5369463])
    H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
                    [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
                    [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])

    paretoParams =  [1.90394177, 2.00408536]

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        sampleS = stats.lomax.rvs(paretoParams[0], size = (n, s, g), random_state = rng) + 1
        sampleD = stats.lomax.rvs(paretoParams[1], size = (n, c, g), random_state = rng) + 1
        return np.concatenate((sampleS, sampleD), axis = 1)
    
    rngEval = np.random.default_rng(seed = 777)
    evalSample = sampler(1000000, -1, rngEval)

    network = BaseNetwork(C, Q_sp, Q_pc, R, M, H)
    
    def evaluator(learningResult: NDArray, repIdx: int) -> float:
        return np.mean(network.objective(learningResult, evalSample))

    sampleSizeList = [2**i for i in range(7, 12)]
    kList = [(10, 0.005)]
    # kList = [(10, 0.1)]
    BList = [200]
    k12List = [((10, 0.005), (10, 0.005))]
    k12List = [((10, 0.1), (10, 0.1))]
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
             numParallelLearn = 10, 
             numParallelEval = 10)