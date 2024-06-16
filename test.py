from BaggingExamples import ReBaggedLRCoef, ReBaggedLRModel
import numpy as np
from multiprocessing import set_start_method
import os
import time



if __name__ == "__main__":
    rngData = np.random.default_rng(seed = 888)
    N = 10000
    d = 10
    XX = rngData.normal(size = (N, d))
    beta = np.linspace(0, 1, num = d)
    y: np.ndarray = np.dot(XX, beta) + rngData.normal(size = N)

    sample = np.hstack((y.reshape(-1, 1), XX))

    if os.name == "posix":
        set_start_method("fork")

    lr = ReBaggedLRCoef(False, numParallelTrain = 4, numParallelEval = 4, randomState = 666)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200)
    print(f"taking {time.time() - tic} secs")
    print(output)

    lr = ReBaggedLRModel(False, numParallelTrain = 4, numParallelEval = 4, randomState = 666)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200)
    print(f"taking {time.time() - tic} secs")
    print(output.coef_)