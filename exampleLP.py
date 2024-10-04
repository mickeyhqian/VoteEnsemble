from VoteEnsemble import MoVE, ROVE, BaseLearner
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
import time



# Linear program with stochastic objective
# min E[\xi^T * x]
# s.t. A * x <= b
#      l <= x <= u

class BaseLP(BaseLearner):
    def __init__(self, A: NDArray, b: NDArray, lb: NDArray, ub: NDArray):
        self._A: NDArray = np.asarray(A)
        self._b: NDArray = np.asarray(b)
        self._lb: NDArray = np.asarray(lb)
        self._ub: NDArray = np.asarray(ub)
    
    def learn(self, sample: NDArray) -> NDArray:
        x = cp.Variable(len(self._lb))
        prob = cp.Problem(cp.Minimize(np.mean(sample, axis = 0) @ x),
                          [self._A @ x <= self._b, x >= self._lb, x <= self._ub])
        prob.solve()

        if prob.status == "optimal":
            return x.value
    
    @property
    def enableDeduplication(self):
        return True
    
    def isDuplicate(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        return np.dot(sample, learningResult)

    @property
    def isMinimization(self):
        return True
    

if __name__ == "__main__":
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

    moveLP = MoVE(lp, randomState = 666, numParallelLearn = 4)
    tic = time.time()
    output = moveLP.run(sample, 500, 200)
    print(f"{MoVE.__name__} outputs the solution: ", output)

    roveLP = ROVE(lp, False, randomState = 666, numParallelLearn = 4)
    tic = time.time()
    output = roveLP.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__} outputs the solution: ", output)

    rovesLP = ROVE(lp, True, randomState = 666, numParallelLearn = 4)
    tic = time.time()
    output = rovesLP.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__}s outputs the solution: ", output)