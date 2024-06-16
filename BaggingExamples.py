from typing import Any
from numpy.typing import NDArray
from Bagging import BAG, ReBAG
from sklearn.linear_model import LinearRegression
import numpy as np


class ReBaggedLRCoef(ReBAG):
    def train(self, sample: NDArray) -> Any:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr.coef_
    
    def identicalTrainingOuputs(self, output1: Any, output2: Any) -> bool:
        return np.max(np.abs(output1 - output2)) < 1e-6
    
    @property
    def isMinimization(self):
        return True
    
    def evaluate(self, trainingOutput: Any, sample: NDArray) -> float:
        error = np.dot(sample[:, 1:], trainingOutput) - sample[:, 0]
        return np.mean(error ** 2)
    

class ReBaggedLRModel(ReBAG):
    def train(self, sample: NDArray) -> Any:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr
    
    def identicalTrainingOuputs(self, output1: LinearRegression, output2: LinearRegression) -> bool:
        return np.max(np.abs(output1.coef_ - output2.coef_)) < 1e-6
    
    @property
    def isMinimization(self):
        return True
    
    def evaluate(self, trainingOutput: LinearRegression, sample: NDArray) -> float:
        error = trainingOutput.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)