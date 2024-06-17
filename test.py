from Examples import BagBinary, BagLP, ReBagLR
import numpy as np
from multiprocessing import set_start_method
import os
import time



if __name__ == "__main__":
    if os.name == "posix":
        set_start_method("fork")
        
    rngData = np.random.default_rng(seed = 888)

    binary = BagBinary(numParallelTrain = 4, randomState = 666)
    sample = binary.genSample(10000, rngData)
    tic = time.time()
    output = binary.run(sample, 1000, 200)
    print(f"BaggedBinary took {time.time() - tic} secs, result: ", output)

    lr = ReBagLR(False, numParallelTrain = 4, numParallelEval = 4, randomState = 666)
    sample = lr.genSample(10000, rngData)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200)
    print(f"ReBaggedLR took {time.time() - tic} secs, result: ", output.coef_)

    d = 10
    lb = np.zeros(d)
    ub = np.full(d, np.inf)
    rngLP = np.random.default_rng(seed=999)
    A = rngLP.uniform(low=0, high=1, size=(20, d))
    b = 10 * np.ones(20)
    c = rngLP.uniform(low=0, high=1, size=d)
    lp = BagLP(c, A, b, lb, ub, numParallelTrain = 4, randomState = 666)
    sample = lp.genSample(10000, rngData)
    tic = time.time()
    output = lp.run(sample, 1000, 200)
    print(f"BaggedLP took {time.time() - tic} secs, result: ", output)