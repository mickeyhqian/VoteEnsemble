from Bagging import BaseTrainer
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List

    

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

    def objective(self, trainingResult: NDArray, sample: NDArray) -> float:
        return np.dot(np.mean(sample, axis = 0), trainingResult)

    @property
    def isMinimization(self):
        return True


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
    
    def objective(self, trainingResult: LinearRegression, sample: NDArray) -> float:
        error = trainingResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2)

    @property
    def isMinimization(self):
        return True
    

class BaseRidge(BaseTrainer):
    def __init__(self, alpha: float):
        self._alpha: float = alpha

    def train(self, sample: NDArray) -> Ridge:
        y = sample[:,0]
        X = sample[:,1:]
        lr = Ridge(alpha = self._alpha, fit_intercept = False, random_state = 666)
        lr.fit(X, y)
        return lr

    @property
    def enableDeduplication(self):
        return False
    
    def isDuplicate(self):
        pass
    
    def objective(self, trainingResult: Ridge, sample: NDArray) -> float:
        error = trainingResult.predict(sample[:, 1:]) - sample[:, 0]
        return np.mean(error ** 2) + np.mean(trainingResult.coef_ ** 2) * self._alpha / len(sample)
    
    @property
    def isMinimization(self):
        return True


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

        problem = cp.Problem(objective, constraints)

        problem.solve(solver="SCS")

        if problem.status == "optimal":
            return x.value

    @property
    def enableDeduplication(self):
        return False

    def isDuplicate(self):
        pass

    def objective(self, trainingResult: NDArray, sample: NDArray) -> float:
        sampleCentered = sample - self._mu.reshape(1, -1)
        covMatrix = np.dot(sampleCentered.T, sampleCentered) / len(sample)
        return np.dot(np.dot(covMatrix, trainingResult), trainingResult)

    @property
    def isMinimization(self):
        return True
    

class RegressionNN(nn.Module):
    def __init__(self, inputSize: int, layerSizes: List[int]):
        super().__init__()
        layers = []
        prevSize = inputSize
        for size in layerSizes:
            layers.append(nn.Linear(prevSize, size))
            layers.append(nn.ReLU())
            prevSize = size

        layers.append(nn.Linear(prevSize, 1))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BaseNN(BaseTrainer):
    def __init__(self, layerSizes: List[int], batchSize: int = 64, minEpochs: int = 5, maxEpochs: int = 30, learningRate: float = 1e-3, useGPU: bool = True):
        self._layerSizes: List[int] = layerSizes
        self._batchSize: int = batchSize
        self._minEpochs: int = max(1, minEpochs)
        self._maxEpochs: int = max(1, maxEpochs)
        self._learningRate: float = learningRate
        self._device: torch.device = torch.device("cuda" if useGPU and torch.cuda.is_available() else "cpu")
        self._cpu: torch.device = torch.device("cpu")

    def train(self, sample: NDArray) -> RegressionNN:
        torch.manual_seed(1109)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = RegressionNN(sample.shape[1] - 1, self._layerSizes).to(self._device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = self._learningRate)

        tensorX = torch.Tensor(sample[:, 1:]).to(self._device)
        tensorY = torch.Tensor(sample[:, :1]).to(self._device)
        dataset = TensorDataset(tensorX, tensorY)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = self._batchSize, shuffle = True)  # Create DataLoader

        minSize = 16384
        maxSize = 131072
        numEpochs = np.log(len(sample) / minSize) * (self._minEpochs - self._maxEpochs) / np.log(maxSize / minSize) + self._maxEpochs
        numEpochs = max(min(int(numEpochs), self._maxEpochs), self._minEpochs)

        model.train()
        for _ in range(numEpochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.to(self._cpu)
        return model

    @property
    def enableDeduplication(self):
        return False

    def isDuplicate(self):
        pass

    def objective(self, trainingResult: RegressionNN, sample: NDArray) -> float:
        trainingResult.to(self._device)
        trainingResult.eval()

        tensorX = torch.Tensor(sample[:, 1:])
        tensorY = torch.Tensor(sample[:, :1]).to(self._device)
        dataset = TensorDataset(tensorX)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        YpredList = []
        with torch.no_grad():
            for data in dataloader:
                data = data[0].to(self._device)
                YpredList.append(trainingResult(data))

        Ypred = torch.cat(YpredList)
        criterion = nn.MSELoss()
        loss = criterion(Ypred, tensorY)

        trainingResult.to(self._cpu)
        return loss.item()

    @property
    def isMinimization(self):
        return True