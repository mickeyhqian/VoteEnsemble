from BaggingExamples import ReBaggedLRCoef, ReBaggedLRModel
import numpy as np
import time

if __name__ == "__main__":
    rngData = np.random.default_rng(seed=668)
    N = 10000
    d = 10
    XX = rngData.normal(size=(N, d))
    beta = np.linspace(0, 1, num=d)
    y: np.ndarray = np.dot(XX, beta) + rngData.normal(size=N)

    sample = np.hstack((y.reshape(-1, 1), XX))

    lr = ReBaggedLRCoef(False, numParallelTrain=10, numParallelEval=10, randomState=666)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200, epsilon=-1)
    print(f"taking {time.time() - tic} secs")
    print(output)

    lr = ReBaggedLRModel(False, numParallelTrain=10, numParallelEval=10, randomState=666)
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200, epsilon=-1)
    print(f"taking {time.time() - tic} secs")
    print(output.coef_)