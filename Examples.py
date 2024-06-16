from Bagging import BAG, ReBAG
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.typing import NDArray



class BaggedBinary(BAG):
    def train(self, sample: NDArray) -> int:
        meanArray = np.mean(sample, axis = 0)
        return np.argmin(meanArray)
    
    def identicalTrainingResults(self, result1: int, result2: int) -> bool:
        return result1 == result2
    

class ReBaggedLRCoef(ReBAG):
    def train(self, sample: NDArray) -> NDArray:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr.coef_
    
    def identicalTrainingResults(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6
    
    @property
    def isMinimization(self):
        return True
    
    def evaluate(self, trainingResult: NDArray, sample: NDArray) -> float:
        error = np.dot(sample[:, 1:], trainingResult) - sample[:, 0]
        return np.mean(error ** 2)
    

class ReBaggedLRModel(ReBAG):
    def train(self, sample: NDArray) -> LinearRegression:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr
    
    def identicalTrainingResults(self, result1: LinearRegression, result2: LinearRegression) -> bool:
        return np.max(np.abs(result1.coef_ - result2.coef_)) < 1e-6
    
    @property
    def isMinimization(self):
        return True
    
    def evaluate(self, trainingResult: LinearRegression, sample: NDArray) -> float:
        error = trainingResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)