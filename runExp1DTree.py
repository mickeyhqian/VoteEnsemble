from BaseLearners import BaseTree, BaseRF, BaseGBDT, BaseXGB
from sklearn.tree import DecisionTreeRegressor
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
    resultDir = sys.argv[1]

    os.makedirs(resultDir, exist_ok = True)
    logger.setLevel(logging.DEBUG)
    logHandler = logging.FileHandler(os.path.join(resultDir, "exp.log"))
    formatter = logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(message)s")
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    rngEval = np.random.default_rng(seed = 777)

    d = 1
    meanX = np.linspace(1, 100, num = d)
    noiseShape = float(sys.argv[2])
    
    algo = sys.argv[3]
    
    logger.info(f"Pareto shape = {noiseShape}")

    def trueMapping(x: NDArray) -> float:
        return 10 * np.sin(x[0]*2*np.pi)

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        XSample = rng.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        noise: NDArray = stats.lomax.rvs(noiseShape, size = n, random_state = rng) \
            - stats.lomax.rvs(noiseShape, size = n, random_state = rng)
        # noise: NDArray = rng.normal(size = n)
        YSample = np.asarray([[trueMapping(x)] for x in XSample]) + noise.reshape(-1, 1)
        return np.hstack((YSample, XSample))
    
    def evalSampler(n: int) -> NDArray:
        XSample = rngEval.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        YSample = np.asarray([[trueMapping(x)] for x in XSample])
        return np.hstack((YSample, XSample))
    
    if algo == "VE":
        baseTree = BaseTree(minSplit=30, minLeaf=10)
    elif algo == "RF":
        baseTree = BaseRF(30, minSplit=30, minLeaf=10, subsampleRatio=0.1)
    elif algo == "GBDT":
        baseTree = BaseGBDT(300, minSplit=30, minLeaf=10, subsampleRatio=0.1)
    elif algo == "XGB":
        baseTree = BaseXGB(300, subsampleRatio=0.1)
    else:
        raise ValueError()

    evalSample = evalSampler(1000000)
    

    def evaluator(learningResult: DecisionTreeRegressor, repIndex: int) -> float:
        return baseTree.objective(learningResult, evalSample).mean()
    
    def inference(learningResult: DecisionTreeRegressor, repIndex: int) -> NDArray[np.float64]:
        return learningResult.predict(evalSample[:, 1:])

    def loss(prediction: NDArray[np.float64], repIndex: int) -> float:
        return np.mean((prediction - evalSample[:, 0])**2)

    sampleSizeList = [2**i for i in range(11, 16, 2)]
    kList = []
    BList = []
    if algo == "VE":
        k12List = [((30, 0.1), (30, 0.005))]
        B12List = [(30, 200)]
    else:
        k12List = []
        B12List = []
    numReplicates = 200

    
    pipeline(resultDir,
             baseTree, 
             sampler, 
             evaluator, 
             inference,
             loss,
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