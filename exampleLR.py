from VoteEnsemble import ROVE, BaseLearner
import numpy as np
from numpy.typing import NDArray



# Linear regression
# min E[(y - x * \beta)^2]

class BaseLR(BaseLearner):
    def learn(self, sample: NDArray) -> NDArray:
        y = sample[:,0]
        X = sample[:,1:]
        gram = np.dot(X.T, X)
        try:
            return np.linalg.solve(gram, np.dot(X.T, y))
        except:
            return None

    @property
    def enableDeduplication(self):
        return False
    
    def isDuplicate(self):
        pass
    
    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        error = np.dot(sample[:, 1:], learningResult) - sample[:, 0]
        return error ** 2

    @property
    def isMinimization(self):
        return True


if __name__ == "__main__":
    rngData = np.random.default_rng(seed = 888)

    lr = BaseLR()
    
    d = 10
    beta = np.linspace(0, 9, num = d)
    dataX = rngData.normal(size = (10000, d))
    dataY = (np.dot(dataX, beta) + rngData.normal(size = len(dataX))).reshape(-1, 1)
    sample = np.hstack((dataY, dataX))

    # skip MoVE as lr.enableDeduplication() == False
    
    print(f"True model parameters = {beta}")
    
    roveLR = ROVE(lr, False, randomState = 666)
    output = roveLR.run(sample)
    print(f"{ROVE.__name__} outputs the parameters: {output}")

    rovesLR = ROVE(lr, True, randomState = 666)
    output = rovesLR.run(sample)
    print(f"{ROVE.__name__}s outputs the parameters: {output}")