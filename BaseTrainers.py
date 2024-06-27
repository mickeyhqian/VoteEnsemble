from Bagging import BaseTrainer
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp

    

class BaseLP(BaseTrainer):
    def __init__(self, A: NDArray, b: NDArray, lb: NDArray, ub: NDArray):
        self._A: NDArray = np.asarray(A)
        self._b: NDArray = np.asarray(b)
        self._lb: NDArray = np.asarray(lb)
        self._ub: NDArray = np.asarray(ub)
    
    def train(self, sample: NDArray) -> NDArray:
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

    @property
    def isMinimization(self):
        return True

    def objective(self, trainingResult: NDArray, sample: NDArray) -> float:
        return np.dot(np.mean(sample, axis = 0), trainingResult)


class BaseLR(BaseTrainer):
    def train(self, sample: NDArray) -> LinearRegression:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr

    @property
    def enableDeduplication(self):
        return False
    
    def isDuplicate(self):
        pass
    
    @property
    def isMinimization(self):
        return True
    
    def objective(self, trainingResult: LinearRegression, sample: NDArray) -> float:
        error = trainingResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)


class BasePortfolio(BaseTrainer):
    def __init__(self, mu: NDArray, b: float):
        self._mu: NDArray = mu
        self._b: float = b

    def train(self, sample: NDArray) -> NDArray:
        k, m = sample.shape
        sampleCentered = sample - self._mu.reshape(1, -1)
        covMatrix = np.dot(sampleCentered.T, sampleCentered) / k

        x = cp.Variable(m)
        objective = cp.Minimize(cp.quad_form(x, covMatrix))
        constraints = [self._mu @ x >= self._b, cp.sum(x) == 1, x >= 0]

        # Formulate the problem
        problem = cp.Problem(objective, constraints)

        problem.solve(solver="SCS")

        if problem.status == "optimal":
            return x.value

    @property
    def enableDeduplication(self):
        return False

    def isDuplicate(self):
        pass

    @property
    def isMinimization(self):
        return True

    def objective(self, trainingResult: NDArray, sample: NDArray) -> float:
        try:
            sampleCentered = sample - self._mu.reshape(1, -1)
            covMatrix = np.dot(sampleCentered.T, sampleCentered) / len(sample)
            return np.dot(np.dot(covMatrix, trainingResult), trainingResult)
        except:
            return None