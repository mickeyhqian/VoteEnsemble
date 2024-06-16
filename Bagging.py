from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Queue
from typing import List, Any, Tuple, Union



class BAG(metaclass = ABCMeta):
    def __init__(self, numParallelTrain: int = 1, randomState: Union[np.random.Generator, int, None] = None):
        """
        numParallelTrain: number of processes used for parallel training, <= 1 disables parallel training. Default 1
        randomState: a random number generator or a seed to be used to initialize a generator. Default None
        """
        if isinstance(randomState, np.random.Generator):
            self._rng = randomState
        elif isinstance(randomState, int) and randomState >= 0:
            self._rng = np.random.default_rng(seed = randomState)
        else:
            self._rng = np.random.default_rng()
        self._numParallelTrain: int = max(1, numParallelTrain)

    @staticmethod
    @abstractmethod
    def train(sample: NDArray) -> Any:
        """
        base training algorithm

        sample: numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point

        return a training result, e.g., a solution vector (for optimization problems) or a machine learning model (for machine learning problems)
        """
        pass

    @staticmethod
    @abstractmethod
    def identicalTrainingOuputs(output1: Any, output2: Any) -> bool:
        """
        return whether two training results are considered identical
        """
        pass

    @staticmethod
    def toPickleable(trainingOutput: Any) -> Any:
        """
        method that transforms a training result to a pickleable object (e.g. basic python types), to be used only if parallel training is enabled (self._numParallelTrain > 1)

        The default implementation directly returns trainingOutput, and is to be overridden if trainingOutput is not pickleable
        """
        return trainingOutput

    @staticmethod
    def fromPickleable(pickleableTrainingOutput: Any) -> Any:
        """
        the inverse of toPickleable, to be used only if parallel training is enabled (self._numParallelTrain > 1)

        Similar to toPickleable, the default implementation directly returns pickleableTrainingOutput, and is to be overridden if the original trainingOutput is not pickleable
        """
        return pickleableTrainingOutput

    def _subProcessTrain(self, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        for index, subsampleIndices in subsampleList:
            trainingOutput = self.train(sample[subsampleIndices])
            if trainingOutput is None:
                queue.put((index, trainingOutput))
            else:
                queue.put((index, self.toPickleable(trainingOutput)))

    def _trainOnSubsamples(self, sample: NDArray, k: int, B: int) -> List:
        if B <= 0:
            raise ValueError(f"{self._trainOnSubsamples.__qualname__}: B = {B} <= 0")
        n: int = len(sample)
        if n < k:
            raise ValueError(f"{self._trainOnSubsamples.__qualname__}: n = {n} < k = {k}")
        
        subsampleLists: List[List[Tuple[int, List[int]]]] = []
        processIndex = 0
        for b in range(B):
            if processIndex >= len(subsampleLists):
                subsampleLists.append([])
            newSubsample = self._rng.choice(n, k, replace=False)
            subsampleLists[processIndex].append((b, newSubsample.tolist()))
            processIndex = (processIndex + 1) % self._numParallelTrain

        trainingOutputList: List = [None for _ in range(B)]

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    trainingOutputList[index] = self.train(sample[subsampleIndices])
        else:
            queue = Queue()
            processList: List[Process] = [Process(target = self._subProcessTrain, args = (sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, trainingOutput = queue.get()
                if trainingOutput is not None:
                    trainingOutputList[index] = self.fromPickleable(trainingOutput)

            for process in processList:
                process.join()

        return [entry for entry in trainingOutputList if entry is not None]

    def _majorityVote(self, trainingOutputList: List) -> Any:
        if len(trainingOutputList) == 0:
            raise ValueError(f"{self._majorityVote.__qualname__}: empty candidate set")
        
        candidateCount: List[int] = [0 for _ in range(len(trainingOutputList))]
        maxIndex = -1
        maxCount = -1
        for i in range(len(trainingOutputList)):
            index = i
            for j in range(i):
                if candidateCount[j] > 0:
                    if self.identicalTrainingOuputs(trainingOutputList[i], trainingOutputList[j]):
                        index = j
                        break
            
            candidateCount[index] += 1
            if candidateCount[index] > maxCount:
                maxIndex = index
                maxCount = candidateCount[index]
        
        return trainingOutputList[maxIndex]
        
    def run(self, sample: NDArray, k: int, B: int) -> Any:
        """
        run BAG

        sample: numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point
        k: subsample size
        B: number of subsamples to draw

        return the bagged training result
        """
        trainingOutputs = self._trainOnSubsamples(np.asarray(sample), k, B)
        return self._majorityVote(trainingOutputs)


class ReBAG(BAG):
    def __init__(self, dataSplit: bool, numParallelEval: int = 1, numParallelTrain: int = 1, randomState: Union[np.random.Generator, int, None] = None):
        """
        dataSplit: whether or not (ReBAGS vs ReBAG) to split the data across the model candidate retrieval phase and the majority-vote phase
        numParallelEval: number of processes used for parallel evaluation of training objective, <= 1 disables parallel evaluation. Default 1
        numParallelTrain: number of processes used for parallel training, <= 1 disables parallel training. Default 1
        randomState: a random number generator or a seed to be used to initialize a generator. Default None
        """
        super().__init__(numParallelTrain = numParallelTrain, randomState = randomState)
        self._dataSplit: bool = dataSplit
        self._numParallelEval: int = max(1, numParallelEval)

    @property
    @abstractmethod
    def isMinimize(self):
        """
        whether the training problem is a minimization
        """
        pass

    @staticmethod
    @abstractmethod
    def evaluate(trainingOutput: Any, sample: NDArray) -> float:
        """
        evaluate the training objective for a training result on a data set (must be consistent with the training objective optimized by self.train)

        trainingOutput: a training result, e.g., a solution vector (for optimization problems) or a machine learning model (for machine learning problems)
        sample: numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point

        return the training objective value
        """
        pass

    def _subProcessEvaluate(self, candidateList: List, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        candidateList = [self.fromPickleable(candidate) for candidate in candidateList]
        for index, subsampleIndices in subsampleList:
            objectiveList: List[float] = [self.evaluate(candidate, sample[subsampleIndices]) for candidate in candidateList]
            queue.put((index, objectiveList))

    def _evaluateOnSubsamples(self, candidateList: List, sample: NDArray, k: int, B: int) -> NDArray:
        if B <= 0:
            raise ValueError(f"{self._evaluateOnSubsamples.__qualname__}: B = {B} <= 0")
        n: int = len(sample)
        if n < k:
            raise ValueError(f"{self._evaluateOnSubsamples.__qualname__}: n = {n} < k = {k}")
        
        subsampleLists: List[List[Tuple[int, List[int]]]] = []
        processIndex = 0
        for b in range(B):
            if processIndex >= len(subsampleLists):
                subsampleLists.append([])
            newSubsample = self._rng.choice(n, k, replace=False)
            subsampleLists[processIndex].append((b, newSubsample.tolist()))
            processIndex = (processIndex + 1) % self._numParallelEval

        evalOutputList: List[List[float]] = [None for _ in range(B)]

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    objectiveList: List[float] = [self.evaluate(candidate, sample[subsampleIndices]) for candidate in candidateList]
                    evalOutputList[index] = objectiveList
        else:
            queue = Queue()
            pickleableList = [self.toPickleable(candidate) for candidate in candidateList]
            processList: List[Process] = [Process(target = self._subProcessEvaluate, args = (pickleableList, sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, evalOutput = queue.get()
                evalOutputList[index] = evalOutput

            for process in processList:
                process.join()

        evalOutputList = np.asarray(evalOutputList, dtype = np.float64)
        if not np.isfinite(evalOutputList).all():
            raise ValueError(f"{self._evaluateOnSubsamples.__qualname__}: failed to evaluate all the training objective values")

        return evalOutputList
    
    @staticmethod
    def _epsilonOptimalProb(gapMatrix: NDArray, epsilon: float) -> NDArray:
        return np.mean(gapMatrix <= epsilon, axis = 0)

    @staticmethod
    def _findEpsilon(gapMatrix: NDArray, autoEpsilonProb: float) -> float:
        probArray = ReBAG._epsilonOptimalProb(gapMatrix, 0)
        if probArray.max() >= autoEpsilonProb:
            return 0
        
        left, right = 0, 1
        probArray = ReBAG._epsilonOptimalProb(gapMatrix, right)
        while probArray.max() < autoEpsilonProb:
            left = right
            right *= 2
            probArray = ReBAG._epsilonOptimalProb(gapMatrix, right)
        
        tolerance = 1e-3
        while max(right - left, (right - left) / (abs(left) / 2 + abs(right) / 2 + 1e-5)) > tolerance:
            mid = (left + right) / 2
            probArray = ReBAG._epsilonOptimalProb(gapMatrix, mid)
            if probArray.max() >= autoEpsilonProb:
                right = mid
            else:
                left = mid
        
        return right
    
    def _gapMatrix(self, evalArray: NDArray) -> NDArray:
        if self.isMinimize:
            bestObj = evalArray.min(axis = 1, keepdims = True)
            gapMatrix = evalArray - bestObj
        else:
            bestObj = evalArray.max(axis = 1, keepdims = True)
            gapMatrix = bestObj - evalArray
        return gapMatrix

    def run(self, sample: NDArray, k1: int, k2: int, B1: int, B2: int, epsilon: float, autoEpsilonProb: float = 0.5) -> Any:
        """
        run ReBAG or ReBAGS

        sample: numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point
        k1: subsample size for the model candidate retrieval phase
        k2: subsample size for the majority-vote phase
        B1: number of subsamples to draw in the model candidate retrieval phase
        B2: number of subsamples to draw in the majority-vote phase
        epsilon: the suboptimality threshold, auto-selection applied if < 0
        autoEpsilonProb: the probability threshold guiding the auto-selection of epsilon. Default 0.5

        return the bagged training result
        """
        sample = np.asarray(sample)
        sample1 = sample
        sample2 = sample

        if self._dataSplit:
            n1 = len(sample) // 2
            if n1 <= 0 or n1 >= len(sample):
                raise ValueError(f"{self.run.__qualname__}: insufficient data n = {len(sample)}")
            sample1 = sample[:n1]
            sample2 = sample[n1:]

        trainingOutputs = self._trainOnSubsamples(sample1, k1, B1)

        retrievedList: List = []
        for output1 in trainingOutputs:
            existing = False
            for output2 in retrievedList:
                if self.identicalTrainingOuputs(output1, output2):
                    existing = True
                    break

            if not existing:
                retrievedList.append(output1)

        if len(retrievedList) == 0:
            raise ValueError(f"{self.run.__qualname__}: failed to retrieve any model")
        
        evalArray = self._evaluateOnSubsamples(retrievedList, sample2, k2, B2)
        gapMatrix = self._gapMatrix(evalArray)

        if epsilon < 0:
            autoEpsilonProb = min(max(0, autoEpsilonProb), 1)
            if self._dataSplit:
                evalArray = self._evaluateOnSubsamples(retrievedList, sample1, k2, B2)
                gapMatrix1 = self._gapMatrix(evalArray)
                epsilon = ReBAG._findEpsilon(gapMatrix1, autoEpsilonProb)
            else:
                epsilon = ReBAG._findEpsilon(gapMatrix, autoEpsilonProb)
    
        probArray = ReBAG._epsilonOptimalProb(gapMatrix, epsilon)
        return retrievedList[np.argmax(probArray)]