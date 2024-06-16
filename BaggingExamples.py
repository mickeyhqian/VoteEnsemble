from typing import Any
from numpy.typing import NDArray
from Bagging import BAG, ReBAG
from sklearn.linear_model import LinearRegression
import numpy as np


class ReBaggedLRCoef(ReBAG):
    @staticmethod
    def train(sample: NDArray) -> Any:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr.coef_
    
    @staticmethod
    def identicalTrainingOuputs(output1: Any, output2: Any) -> bool:
        return np.max(np.abs(output1 - output2)) < 1e-6
    
    @property
    def isMinimize(self):
        return True
    
    @staticmethod
    def evaluate(trainingOutput: Any, sample: NDArray) -> float:
        error = np.dot(sample[:, 1:], trainingOutput) - sample[:, 0]
        return np.mean(error ** 2)
    

class ReBaggedLRModel(ReBAG):
    @staticmethod
    def train(sample: NDArray) -> Any:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr
    
    @staticmethod
    def identicalTrainingOuputs(output1: LinearRegression, output2: LinearRegression) -> bool:
        return np.max(np.abs(output1.coef_ - output2.coef_)) < 1e-6
    
    @property
    def isMinimize(self):
        return True
    
    @staticmethod
    def evaluate(trainingOutput: LinearRegression, sample: NDArray) -> float:
        error = trainingOutput.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)
    
