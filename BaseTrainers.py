from Bagging import BaseTrainer
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Union

    

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
        return np.mean(error ** 2)
    
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
    def __init__(self, layerSizes: List[int], batchSize: int = 64, minEpochs: int = 10, maxEpochs: int = 30, learningRate: float = 1e-3, useGPU: bool = False):
        self._layerSizes: List[int] = layerSizes
        self._batchSize: int = batchSize
        self._minEpochs: int = max(1, minEpochs)
        self._maxEpochs: int = max(1, maxEpochs)
        self._learningRate: float = learningRate
        self._device: torch.device = torch.device("cuda" if useGPU and torch.cuda.is_available() else "cpu")
        self._cpu: torch.device = torch.device("cpu")

    def _evaluate(self, trainingResult: RegressionNN, dataloader: DataLoader, device: torch.device) -> float:
        trainingResult.eval()
        criterion = nn.MSELoss(reduction = "sum")
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = trainingResult(inputs)
                totalLoss += criterion(outputs, targets).item()

        return totalLoss / len(dataloader.dataset)

    def train(self, sample: NDArray) -> RegressionNN:
        torch.manual_seed(1109)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = RegressionNN(sample.shape[1] - 1, self._layerSizes).to(self._device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = self._learningRate)

        trainSize = int(len(sample) * 0.7)
        validSize = len(sample) - trainSize

        if trainSize < 1 or validSize < 1:
            raise ValueError(f"Insufficient data: training set size = {trainSize}, validation set size = {validSize}")

        trainTensorX = torch.Tensor(sample[:trainSize, 1:])
        trainTensorY = torch.Tensor(sample[:trainSize, :1])
        trainDataset = TensorDataset(trainTensorX, trainTensorY)  # Create dataset
        trainDataLoader = DataLoader(trainDataset, batch_size = self._batchSize, shuffle = True)  # Create DataLoader
        # evalDataLoader = DataLoader(trainDataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        validTensorX = torch.Tensor(sample[trainSize:, 1:])
        validTensorY = torch.Tensor(sample[trainSize:, :1])
        validDataset = TensorDataset(validTensorX, validTensorY)  # Create dataset
        validDataLoader = DataLoader(validDataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        minSize = 16384
        maxSize = 131072
        numEpochs = np.log(trainSize / minSize) * (self._minEpochs - self._maxEpochs) / np.log(maxSize / minSize) + self._maxEpochs
        numEpochs = max(min(int(numEpochs), self._maxEpochs), self._minEpochs)

        bestValidLoss = float("inf")
        numStall = 0
        for i in range(numEpochs):
            # trainLoss = self._evaluate(model, evalDataLoader, self._device)
            validLoss = self._evaluate(model, validDataLoader, self._device)
            if i == 0 or validLoss < bestValidLoss * 0.97:
                bestValidLoss = validLoss
                numStall = 0
            else:
                numStall += 1
            # logger.info(f"#epochs = {i}, training loss = {trainLoss}, validation loss = {validLoss}, best validation loss = {bestValidLoss}, #stall = {numStall}")

            if numStall >= 3:
                # logger.info("early stopped due to #stall")
                break

            model.train()
            for inputs, targets in trainDataLoader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.to(self._cpu)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return model

    @property
    def enableDeduplication(self):
        return False

    def isDuplicate(self):
        pass

    def objective(self, trainingResult: RegressionNN, sample: NDArray, device: Union[torch.device, None] = None) -> float:
        if device is None:
            device = self._device

        trainingResult.to(device)

        tensorX = torch.Tensor(sample[:, 1:])
        tensorY = torch.Tensor(sample[:, :1])
        dataset = TensorDataset(tensorX, tensorY)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        obj = self._evaluate(trainingResult, dataloader, device)

        trainingResult.to(self._cpu)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return obj

    @property
    def isMinimization(self):
        return True
    
    def inference(self, trainingResult: RegressionNN, sample: NDArray, device: Union[torch.device, None] = None) -> torch.Tensor:
        if device is None:
            device = self._device

        trainingResult.to(device)

        tensorX = torch.Tensor(sample[:, 1:])
        dataset = TensorDataset(tensorX)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        trainingResult.eval()

        YPred = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(device)
                outputs = trainingResult(inputs)
                YPred.append(outputs.to(self._cpu))

        trainingResult.to(self._cpu)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return torch.concat(YPred)