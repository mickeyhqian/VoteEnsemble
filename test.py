from Bagging import BAG, ReBAG
from BaseTrainers import BaseLP
import numpy as np
from multiprocessing import set_start_method
import time



if __name__ == "__main__":
    set_start_method("spawn")
        
    rngData = np.random.default_rng(seed = 888)

    d = 10
    lb = np.zeros(d)
    ub = np.full(d, np.inf)
    rngLP = np.random.default_rng(seed=999)
    A = rngLP.uniform(low=0, high=1, size=(5, d))
    b = 10 * np.ones(5)
    lp = BaseLP(A, b, lb, ub)

    c = -rngLP.uniform(low=0, high=1, size=d)
    sample = rngData.normal(loc = c, size = (10000, len(c)))

    bagLP = BAG(lp, randomState = 666)
    tic = time.time()
    output = bagLP.run(sample, 500, 200)
    print(f"BAG took {time.time() - tic} secs, result: ", output)

    rebagLP = ReBAG(lp, False, randomState = 666)
    tic = time.time()
    output = rebagLP.run(sample, 1000, 500, 50, 200)
    print(f"ReBAG took {time.time() - tic} secs, result: ", output)

    rebagsLP = ReBAG(lp, True, randomState = 666)
    tic = time.time()
    output = rebagsLP.run(sample, 1000, 500, 50, 200)
    print(f"ReBAGS took {time.time() - tic} secs, result: ", output)