from VoteEnsemble import ROVE, BaseLearner
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
import time



# Linear regression
# min E[(y - x * \beta)^2]

class BaseLR(BaseLearner):
    def learn(self, sample: NDArray) -> LinearRegression:
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
    
    def objective(self, learningResult: LinearRegression, sample: NDArray) -> NDArray:
        error = learningResult.predict(sample[:, 1:]) - sample[:, 0]
        return error ** 2

    @property
    def isMinimization(self):
        return True


if __name__ == "__main__":
    rngData = np.random.default_rng(seed = 888)

    d = 10
    beta = np.linspace(0, 9, num = d)
    dataX = rngData.normal(size = (10000, d))
    dataY = (np.dot(dataX, beta) + rngData.normal(size = len(dataX))).reshape(-1, 1)
    sample = np.hstack((dataY, dataX))
    lr = BaseLR()

    # skip MoVE as lr.enableDeduplication() == False
    
    roveLR = ROVE(lr, False, randomState = 666)
    output = roveLR.run(sample)
    print(f"{ROVE.__name__} outputs the parameters: ", output.coef_)

    rovesLR = ROVE(lr, True, randomState = 666)
    output = rovesLR.run(sample)
    print(f"{ROVE.__name__}s outputs the parameters: ", output.coef_)