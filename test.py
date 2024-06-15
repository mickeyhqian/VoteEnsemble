from BaggingExamples import ReBaggedLR
import numpy as np
import time

if __name__ == "__main__":
    rng = np.random.default_rng(seed=666)
    lr = ReBaggedLR(False, numParallelTrain=10, numParallelEval=5, randomState=rng)

    rngData = np.random.default_rng(seed=668)
    N = 10000
    d = 10
    XX = rngData.normal(size=(N, d))
    beta = np.linspace(0, 1, num=d)
    y = np.dot(XX, beta) + rngData.normal(size=N)

    sample = np.hstack((y.reshape(-1, 1), XX))
    tic = time.time()
    output = lr.run(sample, 1000, 1000, 100, 200, epsilon=-1)
    print(f"taking {time.time() - tic} secs")
    print(output)