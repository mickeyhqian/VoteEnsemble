from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Queue
import pickle
import os
from zstandard import ZstdCompressor, ZstdDecompressor
from typing import List, Any, Tuple, Union



class BaseLearner(metaclass = ABCMeta):
    @abstractmethod
    def learn(self, sample: NDArray) -> Any:
        """
        The learning algorithm.

        Args:
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.

        Returns:
            A learning result of any type, e.g., a solution scalar/vector (for optimization problems) or a machine learning model (for machine learning problems).
        """
        pass

    @property
    @abstractmethod
    def enableDeduplication(self):
        """
        Property of whether or not to deduplicate learning results from subsamples using the self.isDuplicate method.

        - Use with MoVE and ROVE:
            - MoVE: MoVE only accepts base learners with self.enableDeduplication = True.
            - ROVE: ROVE accepts any base learner, but setting self.enableDeduplication = True when appropriate can reduce the computation.
            
        - Examples suggested to set self.enableDeduplication = True:
            - Problems with discrete spaces, such as combinatorial/integer optimization.
            - Problems with continuous spaces but learning results selected from a discrete subspace, such as linear programs solved with the simplex method.
        """
        pass

    @abstractmethod
    def isDuplicate(self, result1: Any, result2: Any) -> bool:
        """
        Returns whether two learning results are considered duplicates of each other.
        
        Invoked only if self.enableDeduplication = True, can be arbitrarily defined otherwise.

        Args:
            result1, result2: 
                Each is a learning result output by self.learn.

        Returns:
            True/False.
        """
        pass

    @abstractmethod
    def objective(self, learningResult: Any, sample: NDArray) -> float:
        """
        Evaluates the empirical objective for a learning result on a data set. self.learn may or may not attempt to optimize the same objective.
        
        Invoked in ROVE only. This evaluation is to be used in the voting phase of ROVE.

        Args:
            learningResult: 
                A learning result output by self.learn.
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.

        Returns:
            The empirical objective value.
        """
        pass

    @property
    @abstractmethod
    def isMinimization(self):
        """
        Property of whether or not the objective defined by self.objective is to be minimized (as opposed to being maximized).

        Invoked in ROVE only.
        """
        pass

    def toPickleable(self, learningResult: Any) -> Any:
        """
        Transforms a learning result to a pickleable object (e.g., basic python types).

        Invoked only if parallel learning/evaluation is enabled or subsampleResultsDir is provided in MoVE/ROVE.

        The default implementation directly returns learningResult, and is to be overridden if learningResult is not pickleable.

        Args:
            learningResult: 
                A learning result output by self.learn.

        Returns:
            A pickleable representation of the learning result.
        """
        return learningResult

    def fromPickleable(self, pickleableLearningResult: Any) -> Any:
        """
        The inverse of toPickleable.

        Invoked only if parallel learning/evaluation is enabled or subsampleResultsDir is provided in MoVE/ROVE.

        The default implementation directly returns pickleableLearningResult, and is to be overridden accordingly if toPickleable is overridden.

        Args:
            pickleableLearningResult: 
                A pickleable representation of a learning result output by self.learn.

        Returns:
            A learning result.
        """
        return pickleableLearningResult
    
    def dumpLearningResult(self, learningResult: Any, destFile: str):
        """
        Dumps a learning result to a file.

        Args:
            learningResult: 
                A learning result output by self.learn.
            destFile: 
                Path of a file to dump the learning result into.
        """
        result = pickle.dumps(self.toPickleable(learningResult))
        compressor = ZstdCompressor()
        result = compressor.compress(result)
        with open(destFile, "wb") as f:
            f.write(result)

    def loadLearningResult(self, sourceFile: str) -> Any:
        """
        Loads a learning result from a file.

        Args:
            sourceFile: 
                Path of a file to load the learning result from.

        Returns:
            A learning result.
        """
        with open(sourceFile, "rb") as f:
            result = f.read()
        decompressor = ZstdDecompressor()
        result = decompressor.decompress(result)
        return self.fromPickleable(pickle.loads(result))


class BaseVE(metaclass = ABCMeta):
    def __init__(self, baseLearner: BaseLearner, 
                 numParallelLearn: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None, 
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        """
        Args:
            baseLearner: 
                A base learner of type BaseLearner.
            numParallelLearn: 
                Number of processes used for parallel learning. A value <= 1 disables parallel learning, default 1.
            randomState: 
                A random number generator or a seed to be used to initialize a random number generator. Default None, random initial state.
            subsampleResultsDir: 
                A directory where learning results on subsamples will be dumped to reduce RAM usage. Default None, i.e., all the learning results are kept in RAM.
            deleteSubsampleResults: 
                Whether to delete learning results on subsamples after finishing the algorithm when subsampleResultsDir is not None. Default True.
        """
        if not isinstance(baseLearner, BaseLearner):
            raise ValueError(f"baseLearner must be of type {BaseLearner.__name__}")
        self._baseLearner: BaseLearner = baseLearner
        self._numParallelLearn: int = max(1, int(numParallelLearn))
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

    def _dumpSubsampleResult(self, learningResult: Any, index: int):
        self._baseLearner.dumpLearningResult(learningResult, self._subsampleResultPath(index))

    def _loadSubsampleResult(self, index: int) -> Any:
        return self._baseLearner.loadLearningResult(self._subsampleResultPath(index))

    def _subProcessLearn(self, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        for index, subsampleIndices in subsampleList:
            learningResult = self._baseLearner.learn(sample[subsampleIndices])
            if learningResult is not None:
                if self._subsampleResultsDir is None:
                    learningResult = self._baseLearner.toPickleable(learningResult)
                else:
                    self._dumpSubsampleResult(learningResult, index)
                    learningResult = True
            queue.put((index, learningResult))

    def _learnOnSubsamples(self, sample: NDArray, k: int, B: int) -> List:
        if B <= 0:
            raise ValueError(f"{self._learnOnSubsamples.__qualname__}: B = {B} <= 0")
        n: int = len(sample)
        if n < k:
            raise ValueError(f"{self._learnOnSubsamples.__qualname__}: n = {n} < k = {k}")
        
        subsampleLists: List[List[Tuple[int, List[int]]]] = []
        processIndex = 0
        for b in range(B):
            if processIndex >= len(subsampleLists):
                subsampleLists.append([])
            newSubsample = self._rng.choice(n, k, replace=False)
            subsampleLists[processIndex].append((b, newSubsample.tolist()))
            processIndex = (processIndex + 1) % self._numParallelLearn

        learningResultList: List = [None for _ in range(B)]

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    learningResult = self._baseLearner.learn(sample[subsampleIndices])
                    if learningResult is not None and self._subsampleResultsDir is not None:
                        self._dumpSubsampleResult(learningResult, index)
                        learningResult = index
                    learningResultList[index] = learningResult
        else:
            queue = Queue()
            processList: List[Process] = [Process(target = self._subProcessLearn, args = (sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, learningResult = queue.get()
                if learningResult is not None:
                    if self._subsampleResultsDir is None:
                        learningResultList[index] = self._baseLearner.fromPickleable(learningResult)
                    else:
                        learningResultList[index] = index

            for process in processList:
                process.join()

        return [entry for entry in learningResultList if entry is not None]
        
    @abstractmethod
    def run(self, sample: NDArray) -> Any:
        pass


class MoVE(BaseVE):
    def __init__(self, baseLearner: BaseLearner, 
                 numParallelLearn: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None,
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        super().__init__(baseLearner, 
                         numParallelLearn = numParallelLearn, 
                         randomState = randomState, 
                         subsampleResultsDir = subsampleResultsDir, 
                         deleteSubsampleResults = deleteSubsampleResults)
        if not self._baseLearner.enableDeduplication:
            raise ValueError(f"{self.__class__.__name__} does not accept base learners with enableDeduplication = False")

    def run(self, sample: NDArray, k: int, B: int) -> Any:
        """
        Run MoVE on the base learner.

        Args:
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.
            k: 
                Subsample size.
            B: 
                Number of subsamples to draw.

        Returns:
            A learning result.
        """
        self._prepareSubsampleResultDir()

        learningResults = self._learnOnSubsamples(np.asarray(sample), k, B)

        if len(learningResults) == 0:
            raise ValueError(f"{self.run.__qualname__}: empty candidate set")
        
        indexCountPairs: List[List[int]] = []
        maxIndex = 0
        maxCount = 0
        for i in range(len(learningResults)):
            result1 = learningResults[i]
            if self._subsampleResultsDir is not None:
                result1 = self._loadSubsampleResult(result1)
            index = len(indexCountPairs)
            for j in range(len(indexCountPairs)):
                result2 = learningResults[indexCountPairs[j][0]]
                if self._subsampleResultsDir is not None:
                    result2 = self._loadSubsampleResult(result2)
                if self._baseLearner.isDuplicate(result1, result2):
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
            return learningResults[maxIndex]
        else:
            output = self._loadSubsampleResult(learningResults[maxIndex])
            if self._deleteSubsampleResults:
                for index in learningResults:
                    resultPath = self._subsampleResultPath(index)
                    if os.path.isfile(resultPath):
                        os.remove(resultPath)
            return output


class ROVE(BaseVE):
    def __init__(self, baseLearner: BaseLearner, 
                 dataSplit: bool, 
                 numParallelEval: int = 1, 
                 numParallelLearn: int = 1, 
                 randomState: Union[np.random.Generator, int, None] = None,
                 subsampleResultsDir: Union[str, None] = None, 
                 deleteSubsampleResults: bool = True):
        """
        Args:
            baseLearner: 
                A base learner of type BaseLearner.
            dataSplit: 
                Whether or not (ROVEs vs ROVE) to split the data across the model candidate retrieval phase and the majority-vote phase.
            numParallelEval: 
                Number of processes used for parallel evaluation of baseLearner.objective. A value <= 1 disables parallel evaluation. Default 1.
            numParallelLearn: 
                Number of processes used for parallel learning. A value <= 1 disables parallel learning, default 1.
            randomState: 
                A random number generator or a seed to be used to initialize a random number generator. Default None, random initial state.
            subsampleResultsDir: 
                A directory where learning results on subsamples will be dumped to reduce RAM usage. Default None, i.e., all the learning results are kept in RAM.
            deleteSubsampleResults: 
                Whether to delete learning results on subsamples after finishing the algorithm when subsampleResultsDir is not None. Default True.
        """
        super().__init__(baseLearner, 
                         numParallelLearn = numParallelLearn, 
                         randomState = randomState,
                         subsampleResultsDir = subsampleResultsDir,
                         deleteSubsampleResults = deleteSubsampleResults)
        self._dataSplit: bool = dataSplit
        self._numParallelEval: int = max(1, int(numParallelEval))

    def _subProcessObjective(self, candidateList: List, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        indexToObj = {entry[0]: [] for entry in subsampleList}
        for candidate in candidateList:
            if self._subsampleResultsDir is None:
                candidate = self._baseLearner.fromPickleable(candidate)
            else:
                candidate = self._loadSubsampleResult(candidate)
            for index, subsampleIndices in subsampleList:
                indexToObj[index].append(self._baseLearner.objective(candidate, sample[subsampleIndices]))
        
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
                        evalOutputList[index].append(self._baseLearner.objective(candidate, sample[subsampleIndices]))
        else:
            queue = Queue()
            if self._subsampleResultsDir is None:
                pickleableList = [self._baseLearner.toPickleable(candidate) for candidate in candidateList]
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
            raise ValueError(f"{self._objectiveOnSubsamples.__qualname__}: failed to evaluate all the objective values")

        return evalOutputList
    
    @staticmethod
    def _epsilonOptimalProb(gapMatrix: NDArray, epsilon: float) -> NDArray:
        return np.mean(gapMatrix <= epsilon, axis = 0)

    @staticmethod
    def _findEpsilon(gapMatrix: NDArray, autoEpsilonProb: float) -> float:
        probArray = ROVE._epsilonOptimalProb(gapMatrix, 0)
        if probArray.max() >= autoEpsilonProb:
            return 0
        
        left, right = 0, 1
        probArray = ROVE._epsilonOptimalProb(gapMatrix, right)
        while probArray.max() < autoEpsilonProb:
            left = right
            right *= 2
            probArray = ROVE._epsilonOptimalProb(gapMatrix, right)
        
        tolerance = 1e-3
        while max(right - left, (right - left) / (abs(left) / 2 + abs(right) / 2 + 1e-5)) > tolerance:
            mid = (left + right) / 2
            probArray = ROVE._epsilonOptimalProb(gapMatrix, mid)
            if probArray.max() >= autoEpsilonProb:
                right = mid
            else:
                left = mid
        
        return right
    
    def _gapMatrix(self, evalArray: NDArray) -> NDArray:
        if self._baseLearner.isMinimization:
            bestObj = evalArray.min(axis = 1, keepdims = True)
            gapMatrix = evalArray - bestObj
        else:
            bestObj = evalArray.max(axis = 1, keepdims = True)
            gapMatrix = bestObj - evalArray
        return gapMatrix

    def run(self, sample: NDArray, k1: int, k2: int, B1: int, B2: int, epsilon: float = -1.0, autoEpsilonProb: float = 0.5) -> Any:
        """
        Run ROVE (self._dataSplit = False) or ROVEs (self._dataSplit = True) on the base learner.

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
            A learning result.
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

        learningResults = self._learnOnSubsamples(sample1, k1, B1)

        retrievedList: List = []
        if self._baseLearner.enableDeduplication:
            for i in range(len(learningResults)):
                result1 = learningResults[i]
                if self._subsampleResultsDir is not None:
                    result1 = self._loadSubsampleResult(result1)
                existing = False
                for result2 in retrievedList:
                    if self._subsampleResultsDir is not None:
                        result2 = self._loadSubsampleResult(result2)
                    if self._baseLearner.isDuplicate(result1, result2):
                        existing = True
                        break

                if not existing:
                    retrievedList.append(learningResults[i])
        else:
            retrievedList = learningResults

        if len(retrievedList) == 0:
            raise ValueError(f"{self.run.__qualname__}: failed to retrieve any learning result")
        
        evalArray = self._objectiveOnSubsamples(retrievedList, sample2, k2, B2)
        gapMatrix = self._gapMatrix(evalArray)

        if epsilon < 0:
            autoEpsilonProb = min(max(0, autoEpsilonProb), 1)
            if self._dataSplit:
                evalArray = self._objectiveOnSubsamples(retrievedList, sample1, k2, B2)
                gapMatrix1 = self._gapMatrix(evalArray)
                epsilon = ROVE._findEpsilon(gapMatrix1, autoEpsilonProb)
            else:
                epsilon = ROVE._findEpsilon(gapMatrix, autoEpsilonProb)
    
        probArray = ROVE._epsilonOptimalProb(gapMatrix, epsilon)

        if self._subsampleResultsDir is None:
            return retrievedList[np.argmax(probArray)]
        else:
            output = self._loadSubsampleResult(retrievedList[np.argmax(probArray)])
            if self._deleteSubsampleResults:
                for index in learningResults:
                    resultPath = self._subsampleResultPath(index)
                    if os.path.isfile(resultPath):
                        os.remove(resultPath)
            return output