from VoteEnsemble import ROVE, BaseLearner
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from multiprocessing import set_start_method
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
    
    def objective(self, learningResult: LinearRegression, sample: NDArray) -> float:
        error = learningResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)

    @property
    def isMinimization(self):
        return True


if __name__ == "__main__":
    set_start_method("spawn")
        
    rngData = np.random.default_rng(seed = 888)

    d = 10
    beta = np.linspace(0, 9, num = d)
    dataX = rngData.normal(size = (10000, d))
    dataY = np.dot(dataX, beta).reshape(-1, 1)
    sample = np.hstack((dataY, dataX))
    lr = BaseLR()

    roveLR = ROVE(lr, False, randomState = 666)
    tic = time.time()
    output = roveLR.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__} took {time.time() - tic} secs, result: ", output.coef_)

    rovesLR = ROVE(lr, True, randomState = 666)
    tic = time.time()
    output = rovesLR.run(sample, 1000, 500, 50, 200)
    print(f"{ROVE.__name__}s took {time.time() - tic} secs, result: ", output.coef_)