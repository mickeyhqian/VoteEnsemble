from VoteEnsemble import MoVE, ROVE, BaseLearner
import numpy as np
from numpy.typing import NDArray



# A simple linear program with stochastic objective
# min E[\xi_1*x_1 + \xi_2*x_2]
# s.t. x_1 + x_2 = 1
#      x_1, x_2 >= 0

class BaseLP(BaseLearner):
    def learn(self, sample: NDArray) -> NDArray:
        xiMean = np.mean(sample, axis = 0)
        if xiMean[0] < xiMean[1]:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
    
    @property
    def enableDeduplication(self):
        return True
    
    def isDuplicate(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        return np.dot(sample, learningResult)

    @property
    def isMinimization(self):
        return True
    

if __name__ == "__main__":
    rngData = np.random.default_rng(seed = 888)

    c = [0.0, 0.2]
    
    lp = BaseLP()
    sample = rngData.normal(loc = c, size = (10000, len(c)))
    
    optimalVal = np.dot(c, lp.learn([c]))
    print(f"True optimal objective value = {optimalVal}")

    moveLP = MoVE(lp, randomState = 666)
    output = moveLP.run(sample)
    print(f"{MoVE.__name__} outputs the solution: {output}, objective value = {np.dot(c, output)}")

    roveLP = ROVE(lp, False, randomState = 666)
    output = roveLP.run(sample)
    print(f"{ROVE.__name__} outputs the solution: {output}, objective value = {np.dot(c, output)}")

    rovesLP = ROVE(lp, True, randomState = 666)
    output = rovesLP.run(sample)
    print(f"{ROVE.__name__}s outputs the solution: {output}, objective value = {np.dot(c, output)}")