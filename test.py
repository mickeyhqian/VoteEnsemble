from Examples import BagBinary, ReBagLR
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

    lr = ReBagLR(False, numParallelTrain = 4, numParallelEval = 4, randomState = 666)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200)
    print(f"ReBaggedLR took {time.time() - tic} secs, result: ", output.coef_)

    lr.resetRandomState()
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200)
    print(f"ReBaggedLR took {time.time() - tic} secs, result: ", output.coef_)

    binarySample = np.hstack((rngData.normal(loc = 0.1, size = (N, 1)), rngData.normal(loc = 0.0, size = (N, 1))))
    binary = BagBinary(numParallelTrain = 4, randomState = 666)
    tic = time.time()
    output = binary.run(binarySample, 1000, 200)
    print(f"BaggedBinary took {time.time() - tic} secs, result: ", output)