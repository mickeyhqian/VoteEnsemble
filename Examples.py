from Bagging import BaseTrainer
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
from scipy import stats

    

class BaseLP(BaseTrainer):
    def __init__(self, c: NDArray, A: NDArray, b: NDArray, lb: NDArray, ub: NDArray):
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
    
    def genSample(self, n: int, rngData: np.random.Generator) -> NDArray:
        return rngData.normal(loc = self._c, size = (n, len(self._c)))


class BaseLR(BaseTrainer):
    def __init__(self, meanX: NDArray, beta: NDArray, noiseShape: float):
        self._meanX: NDArray = np.asarray(meanX)
        self._beta: NDArray = np.asarray(beta)
        self._noiseShape: float = noiseShape
        
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
    
    def genSample(self, n: int, rngData: np.random.Generator) -> NDArray:
        XSample = rngData.uniform(low = 0, high = 2 * self._meanX, size = (n, len(self._meanX)))
        
        noise: NDArray = stats.lomax.rvs(self._noiseShape, size = n, random_state = rngData) - stats.lomax.rvs(self._noiseShape, size = n, random_state = rngData)

        YSample = np.dot(XSample, np.reshape(self._beta, (-1,1))) + noise.reshape(-1, 1)

        return np.hstack((YSample, XSample))

    def optimalityGap(self, trainingResult: LinearRegression) -> float:
        error = trainingResult.coef_ - self._beta
        XVars = (2*self._meanX)**2 / 12
        return np.dot(self._meanX, error) ** 2 + np.sum(error**2 * XVars)