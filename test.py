from VoteEnsemble import MoVE, ROVE
from BaseLearners import BaseLP
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

    moveLP = MoVE(lp, randomState = 666)
    tic = time.time()
    output = moveLP.run(sample, 500, 200)
    print(f"{MoVE.__name__} took {time.time() - tic} secs, result: ", output)

    roveLP = ROVE(lp, False, randomState = 666)
    tic = time.time()
    output = roveLP.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__} took {time.time() - tic} secs, result: ", output)

    rovesLP = ROVE(lp, True, randomState = 666)
    tic = time.time()
    output = rovesLP.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__}s took {time.time() - tic} secs, result: ", output)