from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Queue
import pickle
import os
from zstandard import ZstdCompressor, ZstdDecompressor
from typing import List, Any, Tuple, Union



class BaseTrainer(metaclass = ABCMeta):
    @abstractmethod
    def train(self, sample: NDArray) -> Any:
        """
        The training algorithm.

        Args:
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.

        Returns:
            A training result of any type, e.g., a solution scalar/vector (for optimization problems) or a machine learning model (for machine learning problems).
        """
        pass

    @property
    @abstractmethod
    def enableDeduplication(self):
        """
        Property of whether or not to deduplicate training results from subsamples using the self.isDuplicate method.

        - Use with BAG and ReBAG:
            - BAG: BAG only accepts base trainers with self.enableDeduplication = True.
            - ReBAG: ReBAG accepts any base trainer, but setting self.enableDeduplication = True when appropriate can reduce the computation.
            
        - Examples suggested to set self.enableDeduplication = True:
            - Training problems with discrete spaces, such as combinatorial/integer optimization.
            - Training problems with continuous spaces but training results selected from a discrete subspace, such as linear programs solved with the simplex method.
        """
        pass

    @abstractmethod
    def isDuplicate(self, result1: Any, result2: Any) -> bool:
        """
        Returns whether two training results are considered duplicates of each other.
        
        Invoked only if self.enableDeduplication = True, can be arbitrarily defined otherwise.

        Args:
            result1, result2: 
                Each is a training result output by self.train.

        Returns:
            True/False
        """
        pass

    @abstractmethod
    def objective(self, trainingResult: Any, sample: NDArray) -> float:
        """
        Evaluates the training objective for a training result on a data set.
        This evaluation is to be used in the voting phase of ReBAG, and may or may not be the same as the training objective optimized by self.train.

        Invoked in ReBAG only.

        Args:
            trainingResult: 
                A training result output by self.train.
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.

        Returns:
            The training objective value.
        """
        pass

    @property
    @abstractmethod
    def isMinimization(self):
        """
        Property of whether or not the training objective self.objective is to be minimized (as opposed to maximized).

        Invoked in ReBAG only.
        """
        pass

    def toPickleable(self, trainingResult: Any) -> Any:
        """
        Transforms a training result to a pickleable object (e.g., basic python types).

        Invoked only if parallel training/evaluation is enabled or subsampleResultsDir is provided in BAG/ReBAG.

        The default implementation directly returns trainingResult, and is to be overridden if trainingResult is not pickleable.

        Args:
            trainingResult: 
                A training result output by self.train.

        Returns:
            A pickleable representation of the training result.
        """
        return trainingResult

    def fromPickleable(self, pickleableTrainingResult: Any) -> Any:
        """
        The inverse of toPickleable.

        Invoked only if parallel training/evaluation is enabled or subsampleResultsDir is provided in BAG/ReBAG.

        The default implementation directly returns pickleableTrainingResult, and is to be overridden accordingly if toPickleable is overridden.

        Args:
            pickleableTrainingResult: 
                A pickleable representation of a training result output by self.train.

        Returns:
            A training result.
        """
        return pickleableTrainingResult
    
    def dumpTrainingResult(self, trainingResult: Any, destFile: str):
        """
        Dumps a training result to a file.

        Args:
            trainingResult: 
                A training result output by self.train.
            destFile: 
                Path of a file to dump the training result into.
        """
        result = pickle.dumps(self.toPickleable(trainingResult))
        compressor = ZstdCompressor()
        result = compressor.compress(result)
        with open(destFile, "wb") as f:
            f.write(result)

    def loadTrainingResult(self, sourceFile: str) -> Any:
        """
        Loads a training result from a file.

        Args:
            sourceFile: 
                Path of a file to load the training result from.

        Returns:
            A training result.
        """
        with open(sourceFile, "rb") as f:
            result = f.read()
        decompressor = ZstdDecompressor()
        result = decompressor.decompress(result)
        return self.fromPickleable(pickle.loads(result))


class BaseBagging(metaclass = ABCMeta):
    def __init__(self, baseTrainer: BaseTrainer, 
                 numParallelTrain: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None, 
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        """
        Args:
            baseTrainer: 
                A base trainer of type BaseTrainer.
            numParallelTrain: 
                Number of processes used for parallel training. A value <= 1 disables parallel training, default 1.
            randomState: 
                A random number generator or a seed to be used to initialize a random number generator. Default None, random initial state.
            subsampleResultsDir: 
                A directory where training results on subsamples will be dumped to reduce RAM usage. Default None, i.e., all the training results are kept in RAM.
            deleteSubsampleResults: 
                Whether to delete training results on subsamples once bagging finishes when subsampleResultsDir is not None.
        """
        if not isinstance(baseTrainer, BaseTrainer):
            raise ValueError(f"baseTrainer must be of type {BaseTrainer.__name__}")
        self._baseTrainer: BaseTrainer = baseTrainer
        self._numParallelTrain: int = max(1, int(numParallelTrain))
        if isinstance(randomState, np.random.Generator):
            self._rng = randomState
        elif isinstance(randomState, int) and randomState >= 0:
            self._rng = np.random.default_rng(seed = randomState)
        else:
            self._rng = np.random.default_rng()
        self._rngState = self._rng.bit_generator.state
        self._subsampleResultsDir: Union[str, None] = subsampleResultsDir
        self._deleteSubsampleResults: bool = deleteSubsampleResults
    
    def resetRandomState(self):
        """
        Resets the random number generator to its initial state.
        """
        self._rng.bit_generator.state = self._rngState

    def _prepareSubsampleResultDir(self):
        if self._subsampleResultsDir is not None:
            os.makedirs(self._subsampleResultsDir, exist_ok = True)

    def _subsampleResultPath(self, index: int) -> str:
        if self._subsampleResultsDir is None:
            raise RuntimeError("subsampleResultsDir is not provided")
        return os.path.join(self._subsampleResultsDir, f"subsampleResult_{index}")

    def _dumpSubsampleResult(self, trainingResult: Any, index: int):
        self._baseTrainer.dumpTrainingResult(trainingResult, self._subsampleResultPath(index))

    def _loadSubsampleResult(self, index: int) -> Any:
        return self._baseTrainer.loadTrainingResult(self._subsampleResultPath(index))

    def _subProcessTrain(self, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        for index, subsampleIndices in subsampleList:
            trainingResult = self._baseTrainer.train(sample[subsampleIndices])
            if trainingResult is not None:
                if self._subsampleResultsDir is None:
                    trainingResult = self._baseTrainer.toPickleable(trainingResult)
                else:
                    self._dumpSubsampleResult(trainingResult, index)
                    trainingResult = True
            queue.put((index, trainingResult))

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

        trainingResultList: List = [None for _ in range(B)]

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    trainingResult = self._baseTrainer.train(sample[subsampleIndices])
                    if trainingResult is not None and self._subsampleResultsDir is not None:
                        self._dumpSubsampleResult(trainingResult, index)
                        trainingResult = index
                    trainingResultList[index] = trainingResult
        else:
            queue = Queue()
            processList: List[Process] = [Process(target = self._subProcessTrain, args = (sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, trainingResult = queue.get()
                if trainingResult is not None:
                    if self._subsampleResultsDir is None:
                        trainingResultList[index] = self._baseTrainer.fromPickleable(trainingResult)
                    else:
                        trainingResultList[index] = index

            for process in processList:
                process.join()

        return [entry for entry in trainingResultList if entry is not None]
        
    @abstractmethod
    def run(self, sample: NDArray) -> Any:
        """
        Run Bagging on the base trainer.
        """
        pass


class BAG(BaseBagging):
    def __init__(self, baseTrainer: BaseTrainer, 
                 numParallelTrain: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None,
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        super().__init__(baseTrainer, 
                         numParallelTrain = numParallelTrain, 
                         randomState = randomState, 
                         subsampleResultsDir = subsampleResultsDir, 
                         deleteSubsampleResults = deleteSubsampleResults)
        if not self._baseTrainer.enableDeduplication:
            raise ValueError("BAG does not accept base trainers with enableDeduplication = False")

    def run(self, sample: NDArray, k: int, B: int) -> Any:
        """
        Run BAG on the base trainer.

        Args:
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.
            k: 
                Subsample size.
            B: 
                Number of subsamples to draw.

        Returns:
            A bagged training result.
        """
        self._prepareSubsampleResultDir()

        trainingResults = self._trainOnSubsamples(np.asarray(sample), k, B)

        if len(trainingResults) == 0:
            raise ValueError(f"{self.run.__qualname__}: empty candidate set")
        
        indexCountPairs: List[List[int]] = []
        maxIndex = 0
        maxCount = 0
        for i in range(len(trainingResults)):
            result1 = trainingResults[i]
            if self._subsampleResultsDir is not None:
                result1 = self._loadSubsampleResult(result1)
            index = len(indexCountPairs)
            for j in range(len(indexCountPairs)):
                result2 = trainingResults[indexCountPairs[j][0]]
                if self._subsampleResultsDir is not None:
                    result2 = self._loadSubsampleResult(result2)
                if self._baseTrainer.isDuplicate(result1, result2):
                    index = j
                    break
                    
            if index < len(indexCountPairs):
                indexCountPairs[index][1] += 1
            else:
                indexCountPairs.append([i, 1])
            
            if indexCountPairs[index][1] > maxCount:
                maxIndex = indexCountPairs[index][0]
                maxCount = indexCountPairs[index][1]
        
        if self._subsampleResultsDir is None:
            return trainingResults[maxIndex]
        else:
            output = self._loadSubsampleResult(trainingResults[maxIndex])
            if self._deleteSubsampleResults:
                for index in trainingResults:
                    resultPath = self._subsampleResultPath(index)
                    if os.path.isfile(resultPath):
                        os.remove(resultPath)
            return output


class ReBAG(BaseBagging):
    def __init__(self, baseTrainer: BaseTrainer, 
                 dataSplit: bool, 
                 numParallelEval: int = 1, 
                 numParallelTrain: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None,
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        """
        Args:
            baseTrainer: 
                A base trainer of type BaseTrainer.
            dataSplit: 
                Whether or not (ReBAG-S vs ReBAG) to split the data across the model candidate retrieval phase and the majority-vote phase.
            numParallelEval: 
                Number of processes used for parallel evaluation of training objectives. A value <= 1 disables parallel evaluation. Default 1.
            numParallelTrain: 
                Number of processes used for parallel training. A value <= 1 disables parallel training, default 1.
            randomState: 
                A random number generator or a seed to be used to initialize a random number generator. Default None, random initial state.
            subsampleResultsDir: 
                A directory where training results on subsamples will be dumped to reduce RAM usage. Default None, i.e., all the training results are kept in RAM.
            deleteSubsampleResults: 
                Whether to delete training results on subsamples once bagging finishes when subsampleResultsDir is not None. Default True.
        """
        super().__init__(baseTrainer, 
                         numParallelTrain = numParallelTrain, 
                         randomState = randomState,
                         subsampleResultsDir = subsampleResultsDir,
                         deleteSubsampleResults = deleteSubsampleResults)
        self._dataSplit: bool = dataSplit
        self._numParallelEval: int = max(1, int(numParallelEval))

    def _subProcessObjective(self, candidateList: List, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        indexToObj = {entry[0]: [] for entry in subsampleList}
        for candidate in candidateList:
            if self._subsampleResultsDir is None:
                candidate = self._baseTrainer.fromPickleable(candidate)
            else:
                candidate = self._loadSubsampleResult(candidate)
            for index, subsampleIndices in subsampleList:
                indexToObj[index].append(self._baseTrainer.objective(candidate, sample[subsampleIndices]))
        
        for index, objList in indexToObj.items():
            queue.put((index, objList))

    def _objectiveOnSubsamples(self, candidateList: List, sample: NDArray, k: int, B: int) -> NDArray:
        if B <= 0:
            raise ValueError(f"{self._objectiveOnSubsamples.__qualname__}: B = {B} <= 0")
        n = len(sample)
        if n < k:
            raise ValueError(f"{self._objectiveOnSubsamples.__qualname__}: n = {n} < k = {k}")
        
        subsampleLists: List[List[Tuple[int, List[int]]]] = []
        processIndex = 0
        for b in range(B):
            if processIndex >= len(subsampleLists):
                subsampleLists.append([])
            newSubsample = self._rng.choice(n, k, replace=False)
            subsampleLists[processIndex].append((b, newSubsample.tolist()))
            processIndex = (processIndex + 1) % self._numParallelEval

        evalOutputList: List[List[float]] = [[] for _ in range(B)]

        if len(subsampleLists) <= 1:
            for candidate in candidateList:
                if self._subsampleResultsDir is not None:
                    candidate = self._loadSubsampleResult(candidate)
                for subsampleList in subsampleLists:
                    for index, subsampleIndices in subsampleList:
                        evalOutputList[index].append(self._baseTrainer.objective(candidate, sample[subsampleIndices]))
        else:
            queue = Queue()
            if self._subsampleResultsDir is None:
                pickleableList = [self._baseTrainer.toPickleable(candidate) for candidate in candidateList]
            else:
                pickleableList = candidateList
            processList: List[Process] = [Process(target = self._subProcessObjective, args = (pickleableList, sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, evalOutput = queue.get()
                evalOutputList[index] = evalOutput

            for process in processList:
                process.join()

        evalOutputList = np.asarray(evalOutputList, dtype = np.float64)
        if not np.isfinite(evalOutputList).all():
            raise ValueError(f"{self._objectiveOnSubsamples.__qualname__}: failed to evaluate all the training objective values")

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
        if self._baseTrainer.isMinimization:
            bestObj = evalArray.min(axis = 1, keepdims = True)
            gapMatrix = evalArray - bestObj
        else:
            bestObj = evalArray.max(axis = 1, keepdims = True)
            gapMatrix = bestObj - evalArray
        return gapMatrix

    def run(self, sample: NDArray, k1: int, k2: int, B1: int, B2: int, epsilon: float = -1.0, autoEpsilonProb: float = 0.5) -> Any:
        """
        Run ReBAG (self._dataSplit = False) or ReBAG-S (self._dataSplit = True) on the base trainer.

        Args:
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.
            k1: 
                Subsample size for the model candidate retrieval phase.
            k2: 
                Subsample size for the majority-vote phase.
            B1: 
                Number of subsamples to draw in the model candidate retrieval phase.
            B2: 
                Number of subsamples to draw in the majority-vote phase.
            epsilon: 
                The suboptimality threshold. Any value < 0 leads to auto-selection. Default -1.0.
            autoEpsilonProb: 
                The probability threshold guiding the auto-selection of epsilon. Default 0.5.

        Returns:
            A bagged training result.
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

        self._prepareSubsampleResultDir()

        trainingResults = self._trainOnSubsamples(sample1, k1, B1)

        retrievedList: List = []
        if self._baseTrainer.enableDeduplication:
            for i in range(len(trainingResults)):
                result1 = trainingResults[i]
                if self._subsampleResultsDir is not None:
                    result1 = self._loadSubsampleResult(result1)
                existing = False
                for result2 in retrievedList:
                    if self._subsampleResultsDir is not None:
                        result2 = self._loadSubsampleResult(result2)
                    if self._baseTrainer.isDuplicate(result1, result2):
                        existing = True
                        break

                if not existing:
                    retrievedList.append(trainingResults[i])
        else:
            retrievedList = trainingResults

        if len(retrievedList) == 0:
            raise ValueError(f"{self.run.__qualname__}: failed to retrieve any training result")
        
        evalArray = self._objectiveOnSubsamples(retrievedList, sample2, k2, B2)
        gapMatrix = self._gapMatrix(evalArray)

        if epsilon < 0:
            autoEpsilonProb = min(max(0, autoEpsilonProb), 1)
            if self._dataSplit:
                evalArray = self._objectiveOnSubsamples(retrievedList, sample1, k2, B2)
                gapMatrix1 = self._gapMatrix(evalArray)
                epsilon = ReBAG._findEpsilon(gapMatrix1, autoEpsilonProb)
            else:
                epsilon = ReBAG._findEpsilon(gapMatrix, autoEpsilonProb)
    
        probArray = ReBAG._epsilonOptimalProb(gapMatrix, epsilon)

        if self._subsampleResultsDir is None:
            return retrievedList[np.argmax(probArray)]
        else:
            output = self._loadSubsampleResult(retrievedList[np.argmax(probArray)])
            if self._deleteSubsampleResults:
                for index in trainingResults:
                    resultPath = self._subsampleResultPath(index)
                    if os.path.isfile(resultPath):
                        os.remove(resultPath)
            return output