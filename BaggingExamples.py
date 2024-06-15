from typing import Any
from numpy.typing import NDArray
from Bagging import BAG, ReBAG
from sklearn.linear_model import LinearRegression
import numpy as np
import os


class ReBaggedLR(ReBAG):
    def train(self, sample: NDArray) -> Any:
        y = sample[:,0]
        X = sample[:,1:]
        lr = LinearRegression(fit_intercept = False)
        lr.fit(X, y)
        return lr.coef_
    
    def identicalTrainingOuputs(self, output1: Any, output2: Any) -> bool:
        return np.max(np.abs(output1 - output2)) < 1e-6
    
    def toPickleble(self, trainingOutput: Any) -> Any:
        return trainingOutput
    
    def fromPickleable(self, pickleableTrainingOutput: Any) -> Any:
        return pickleableTrainingOutput
    
    @property
    def isMinimize(self):
        return True
    
    def evaluate(self, trainingOutput: Any, sample: NDArray) -> float:
        error = np.dot(sample[:, 1:], trainingOutput) - sample[:, 0]
        return np.mean(error ** 2)
    
