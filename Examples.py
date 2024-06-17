from Bagging import BAG, ReBAG
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
from typing import Union



class BagBinary(BAG):
    def train(self, sample: NDArray) -> int:
        meanArray = np.mean(sample, axis = 0)
        return np.argmin(meanArray)
    
    def isIdentical(self, result1: int, result2: int) -> bool:
        return result1 == result2
    
    def genSample(self, n: int, rng: np.random.Generator) -> NDArray:
        return np.hstack((rng.normal(loc = 0.1, size = (n, 1)), rng.normal(loc = 0.0, size = (n, 1))))
    

class BagLP(BAG):
    def __init__(self, c: NDArray, A: NDArray, b: NDArray, lb: NDArray, ub: NDArray, numParallelTrain: int = 1, randomState: Union[np.random.Generator, int, None] = None):
        super().__init__(numParallelTrain, randomState)
        self._c: NDArray = np.asarray(c)
        self._A: NDArray = np.asarray(A)
        self._b: NDArray = np.asarray(b)
        self._lb: NDArray = np.asarray(lb)
        self._ub: NDArray = np.asarray(ub)
    
    def train(self, sample: NDArray) -> NDArray:
        x = cp.Variable(len(self._c))
        prob = cp.Problem(cp.Minimize(np.mean(sample, axis = 0) @ x),
                          [self._A @ x <= self._b, x >= self._lb, x <= self._ub])
        prob.solve()

        if prob.status == "optimal":
            return x.value
    
    def isIdentical(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6
    
    def genSample(self, n: int, rng: np.random.Generator) -> NDArray:
        return rng.normal(loc = self._c, size = (n, len(self._c)))


class ReBagLR(ReBAG):
    def train(self, sample: NDArray) -> LinearRegression:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr
    
    def isIdentical(self, result1: LinearRegression, result2: LinearRegression) -> bool:
        return np.max(np.abs(result1.coef_ - result2.coef_)) < 1e-6
    
    @property
    def isMinimization(self):
        return True
    
    def evaluate(self, trainingResult: LinearRegression, sample: NDArray) -> float:
        error = trainingResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)
    
    def genSample(self, n: int, rng: np.random.Generator) -> NDArray:
        d = 10
        XX = rng.normal(size = (n, d))
        beta = np.linspace(0, 1, num = d)
        y: NDArray = np.dot(XX, beta) + rng.normal(size = n)
        return np.hstack((y.reshape(-1, 1), XX))