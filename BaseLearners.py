from VoteEnsemble import BaseLearner
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from gurobipy import Model, GRB, quicksum
from typing import List, Union, Dict, Tuple



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
        return (learningResult.predict(sample[:, 1:]) - sample[:, 0]) ** 2

    @property
    def isMinimization(self):
        return True
    

class BaseRidge(BaseLearner):
    def __init__(self, alpha: float):
        self._alpha: float = alpha

    def learn(self, sample: NDArray) -> Ridge:
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
    
    def objective(self, learningResult: Ridge, sample: NDArray) -> NDArray:
        return (learningResult.predict(sample[:, 1:]) - sample[:, 0]) ** 2
    
    @property
    def isMinimization(self):
        return True


class BasePortfolio(BaseLearner):
    def __init__(self, mu: NDArray, b: float):
        self._mu: NDArray = mu
        self._b: float = b

    def learn(self, sample: NDArray) -> Union[NDArray, None]:
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

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        return (np.dot(sample, learningResult) - np.dot(self._mu, learningResult)) ** 2

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


class BaseNN(BaseLearner):
    def __init__(self, layerSizes: List[int], batchSize: int = 64, minEpochs: int = 10, maxEpochs: int = 30, learningRate: float = 1e-3, useGPU: bool = False):
        self._layerSizes: List[int] = layerSizes
        self._batchSize: int = batchSize
        self._minEpochs: int = max(1, minEpochs)
        self._maxEpochs: int = max(1, maxEpochs)
        self._learningRate: float = learningRate
        self._device: torch.device = torch.device("cuda" if useGPU and torch.cuda.is_available() else "cpu")
        self._cpu: torch.device = torch.device("cpu")

    def _evaluate(self, learningResult: RegressionNN, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
        learningResult.eval()
        criterion = nn.MSELoss(reduction = "none")
        lossValues = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = learningResult(inputs)
                newLoss = criterion(outputs, targets)
                lossValues.append(newLoss)

        return torch.cat(lossValues)

    def learn(self, sample: NDArray) -> RegressionNN:
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
            # trainLoss = self._evaluate(model, evalDataLoader, self._device).mean().item()
            validLoss = self._evaluate(model, validDataLoader, self._device).mean().item()
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

    def objective(self, learningResult: RegressionNN, sample: NDArray, device: Union[torch.device, None] = None) -> NDArray:
        if device is None:
            device = self._device

        learningResult.to(device)

        tensorX = torch.Tensor(sample[:, 1:])
        tensorY = torch.Tensor(sample[:, :1])
        dataset = TensorDataset(tensorX, tensorY)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        obj = self._evaluate(learningResult, dataloader, device).to(self._cpu).numpy().flatten()

        learningResult.to(self._cpu)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return obj

    @property
    def isMinimization(self):
        return True
    
    def inference(self, learningResult: RegressionNN, sample: NDArray, device: Union[torch.device, None] = None) -> torch.Tensor:
        if device is None:
            device = self._device

        learningResult.to(device)

        tensorX = torch.Tensor(sample[:, 1:])
        dataset = TensorDataset(tensorX)  # Create dataset
        dataloader = DataLoader(dataset, batch_size = 131072, shuffle = False)  # Create DataLoader

        learningResult.eval()

        YPred = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(device)
                outputs = learningResult(inputs)
                YPred.append(outputs.to(self._cpu))

        learningResult.to(self._cpu)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return torch.concat(YPred)
    
    
class BaseLP(BaseLearner):
    def __init__(self, A: NDArray):
        self._A: NDArray = A.copy()
    
    def learn(self, sample: NDArray) -> Union[NDArray, None]:
        # sample_k: k * N * (N-1)/2 matrix, where each column corresponds the weight of an edge
        # N: number of nodes >= 4.
        # w: dictionary of edge weights, using 1-based index
        # A: some matrices of dimension N * N, used to destroy the total unimodularity, stored as a numpy matrix
        N = len(self._A)
        sampleMean = np.mean(sample, axis = 0)
        edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
        
        model = Model()
        model.setParam(GRB.Param.OutputFlag, 0)
        model.setParam(GRB.Param.Method, 0)
        x = model.addVars(edges, vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
        model.setObjective(quicksum(sampleMean[i] * x[edges[i]] for i in range(len(edges))), GRB.MAXIMIZE)
        model.addConstrs(quicksum(self._A[i - 1, j - 1] * x[(min(i, j), max(i, j))] for j in range(1, N + 1) if i != j) <= 1 for i in range(1, N + 1))
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return np.array([x[e].X for e in edges])
    
    @property
    def enableDeduplication(self):
        return True
    
    def isDuplicate(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        return np.dot(sample, learningResult)

    @property
    def isMinimization(self):
        return False
    
    
class BaseMatching(BaseLearner):
    def __init__(self, w: Dict[Tuple[int, int], float], N: int, randomN: int, epsilon: float):
        self._w: Dict[Tuple[int, int], float] = w.copy()
        self._N: int = N
        self._randomN: int = randomN
        self._epsilon: float = epsilon
    
    def learn(self, sample: NDArray) -> Union[NDArray, None]:
        # maximum weight bipartite matching
        # sample_k: k * 9 matrix, where each column corresponds the weight of an edge
        # N: number of one-side nodes >= 6
        # w: dictionary of edge weights, using 0-based index
        N = self._N
        w = self._w.copy()
        ind = 0
        sampleMean = np.mean(sample, axis = 0)
        for i in range(self._randomN):
            for j in range(self._randomN):
                w[(i,j)] = sampleMean[ind] - self._epsilon
                ind += 1
        
        model = Model()
        model.setParam(GRB.Param.OutputFlag, 0)
        model.setParam(GRB.Param.Method, 0) 
        edges = [(i, j) for i in range(N) for j in range(N)]
        x = model.addVars(edges, vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
        model.setObjective(quicksum(w[e] * x[e] for e in edges), GRB.MAXIMIZE)
        model.addConstrs(quicksum(x[(i,j)] for j in range(N)) <= 1 for i in range(N))
        model.addConstrs(quicksum(x[(i,j)] for i in range(N)) <= 1 for j in range(N))
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return np.array([round(x[e].X) for e in edges], dtype = np.int32)
    
    @property
    def enableDeduplication(self):
        return True
    
    def isDuplicate(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        ind = 0
        randomPart = []
        fixedPart = 0
        for i in range(self._N):
            for j in range(self._N):
                if i < self._randomN and j < self._randomN:
                    randomPart.append(learningResult[ind])
                else:
                    fixedPart += self._w[(i, j)] * learningResult[ind]
                ind += 1
        return np.dot(sample, randomPart) + fixedPart

    @property
    def isMinimization(self):
        return False
    
    
class BaseNetwork(BaseLearner):
    def __init__(self, C: NDArray, Q_sp: NDArray, Q_pc: NDArray, R: NDArray, M: NDArray, H: NDArray):
        self._C: NDArray = C.copy()
        self._Q_sp: NDArray = Q_sp.copy()
        self._Q_pc: NDArray = Q_pc.copy()
        self._R: NDArray = R.copy()
        self._M: NDArray = M.copy()
        self._H: NDArray = H.copy()
    
    def learn(self, sample: NDArray) -> Union[NDArray, None]:
        # input:
        # C (unit cost for building a facility): p*1 numpy array
        # Q_sp (unit flow cost): s*p*g numpy array
        # Q_pc (unit flow cost): p*c*g numpy array
        # sample_S (supply): k*s*g numpy array
        # sample_D (demand): k*c*g numpy array
        # R (unit processing require): p*g numpy array
        # M (processing capacity): p*1 numpy array
        # H (multiplier): c*g numpy array
        # output:
        # x (open facility): p*1 numpy array
        # y_sp (flow supplier --> facility): s*p*g*k numpy array 
        # y_pc (flow facility --> consumer): p*c*g*k numpy array
        # z (multiplier): c*g*k numpy array
        s, p, g = self._Q_sp.shape
        sampleS = sample[:,:s,:]
        sampleD = sample[:,s:,:]
        k, c, _ = sampleD.shape
        model = Model()
        model.setParam(GRB.Param.OutputFlag, 0)
        x = model.addVars(p, vtype = GRB.BINARY)
        y_sp = model.addVars(s, p, g, k, lb = 0, vtype = GRB.CONTINUOUS)
        y_pc = model.addVars(p, c, g, k, lb = 0, vtype = GRB.CONTINUOUS)
        z = model.addVars(c, g, k, lb = 0, vtype = GRB.CONTINUOUS)

        obj_expr = quicksum(self._C[j] * x[j] for j in range(p)) \
                        + 1/k * quicksum(self._Q_sp[i, j, l] * y_sp[i, j, l, a] for i in range(s) for j in range(p) for l in range(g) for a in range(k))\
                            + 1/k * quicksum(self._Q_pc[j, i, l] * y_pc[j, i, l, a] for j in range(p) for i in range(c) for l in range(g) for a in range(k))\
                                + 1/k * quicksum(self._H[i, l] * z[i, l, a] for i in range(c) for l in range(g) for a in range(k))
        
        model.setObjective(obj_expr, GRB.MINIMIZE)

        model.addConstrs((quicksum(y_sp[i, j, l, a] for i in range(s)) - quicksum(y_pc[j, i, l, a] for i in range(c)) == 0 
                        for a in range(k) for l in range(g) for j in range(p)), name="flow")
        
        model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sampleD[a, i, l]
                        for a in range(k) for l in range(g) for i in range(c)), name="demand")
        
        model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sampleS[a, i, l]
                        for a in range(k) for l in range(g) for i in range(s)), name="supply")
        
        model.addConstrs((quicksum(self._R[j, l] * quicksum(y_sp[i, j, l, a] for i in range(s)) for l in range(g)) <= self._M[j] * x[j]
                        for a in range(k) for j in range(p)), name="capacity")

        model.setParam(GRB.Param.Threads, 1)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return np.array([round(x[i].X) for i in range(p)], dtype = np.int32)
    
    @property
    def enableDeduplication(self):
        return True
    
    def isDuplicate(self, result1: NDArray, result2: NDArray) -> bool:
        return np.max(np.abs(result1 - result2)) < 1e-6

    def objective(self, learningResult: NDArray, sample: NDArray) -> NDArray:
        s, p, g = self._Q_sp.shape
        sampleS = sample[:,:s,:]
        sampleD = sample[:,s:,:]
        n, c, _ = sampleD.shape
        model = Model()
        model.setParam(GRB.Param.OutputFlag, 0)
        y_sp = model.addVars(s, p, g, n, lb = 0, vtype = GRB.CONTINUOUS)
        y_pc = model.addVars(p, c, g, n, lb = 0, vtype = GRB.CONTINUOUS)
        z = model.addVars(c, g, n, lb = 0, vtype = GRB.CONTINUOUS)

        obj_expr = 1/n * quicksum(self._Q_sp[i, j, l] * y_sp[i, j, l, a] for i in range(s) for j in range(p) for l in range(g) for a in range(n))\
                        + 1/n * quicksum(self._Q_pc[j, i, l] * y_pc[j, i, l, a] for j in range(p) for i in range(c) for l in range(g) for a in range(n))\
                            + 1/n * quicksum(self._H[i, l] * z[i, l, a] for i in range(c) for l in range(g) for a in range(n))
        
        model.setObjective(obj_expr, GRB.MINIMIZE)

        model.addConstrs((quicksum(y_sp[i, j, l, a] for i in range(s)) - quicksum(y_pc[j, i, l, a] for i in range(c)) == 0
                            for a in range(n) for l in range(g) for j in range(p)), name="flow")
        
        model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sampleD[a, i, l]
                            for a in range(n) for l in range(g) for i in range(c)), name="demand")
        
        model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sampleS[a, i, l]
                            for a in range(n) for l in range(g) for i in range(s)), name="supply")
        
        model.addConstrs((quicksum(self._R[j, l] * quicksum(y_sp[i, j, l, a] for i in range(s)) for l in range(g)) <= self._M[j] * learningResult[j]
                            for a in range(n) for j in range(p)), name="capacity")
        
        model.setParam(GRB.Param.Threads, 1)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            fixedCost = np.dot(self._C, learningResult)
            randomCost1 = np.array([sum(self._Q_sp[i, j, l] * y_sp[i, j, l, a].X for i in range(s) for j in range(p) for l in range(g)) for a in range(n)])
            randomCost2 = np.array([sum(self._Q_pc[j, i, l] * y_pc[j, i, l, a].X for j in range(p) for i in range(c) for l in range(g)) for a in range(n)])
            randomCost3 = np.array([sum(self._H[i, l] * z[i, l, a].X for i in range(c) for l in range(g)) for a in range(n)])
            return randomCost1 + randomCost2 + randomCost3 + fixedCost
        else:
            raise RuntimeError("failed to evaluate network second stage")

    @property
    def isMinimization(self):
        return True