from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing.pool import Pool
import pickle
import os
from zstandard import ZstdCompressor, ZstdDecompressor
from typing import List, Any, Tuple, Union



class BaseLearner(metaclass = ABCMeta):
    """
    The base class for all base learners. Must be pickleable for parallel learning or evaluation.
    """
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
    def objective(self, learningResult: Any, sample: NDArray) -> NDArray:
        """
        Evaluates the objective for a learning result on a data set. self.learn may or may not attempt to optimize the same objective.
        
        Invoked in ROVE only. This evaluation is to be used in the voting phase of ROVE.

        Args:
            learningResult: 
                A learning result output by self.learn.
            sample: 
                A numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point.

        Returns:
            A numpy array consisting of an objective value per data point.
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


class _SubsampleResultIO:
    def __init__(self, baseLearner: BaseLearner, subsampleResultsDir: str):
        self._baseLearner: BaseLearner = baseLearner
        self._subsampleResultsDir: str = subsampleResultsDir
    
    @staticmethod
    def _prepareSubsampleResultDir(subsampleResultsDir: str):
        os.makedirs(subsampleResultsDir, exist_ok = True)

    @staticmethod
    def _subsampleResultPath(subsampleResultsDir: str, index: int) -> str:
        return os.path.join(subsampleResultsDir, f"subsampleResult_{index}")

    @staticmethod
    def _dumpSubsampleResult(baseLearner: BaseLearner, learningResult: Any, subsampleResultsDir: str, index: int):
        baseLearner.dumpLearningResult(learningResult, _SubsampleResultIO._subsampleResultPath(subsampleResultsDir, index))

    @staticmethod
    def _loadSubsampleResult(baseLearner: BaseLearner, subsampleResultsDir: str, index: int) -> Any:
        return baseLearner.loadLearningResult(_SubsampleResultIO._subsampleResultPath(subsampleResultsDir, index))
    
    @staticmethod
    def _deleteSubsampleResults(subsampleResultsDir: str, indexList: List[int]):
        for index in indexList:
            resultPath = _SubsampleResultIO._subsampleResultPath(subsampleResultsDir, index)
            if os.path.isfile(resultPath):
                os.remove(resultPath)


def _subProcessLearn(baseLearner: BaseLearner, 
                     subsampleResultsDir: Union[str, None], 
                     sample: NDArray, 
                     subsampleList: List[Tuple[int, List[int]]]) -> List[Tuple[int, Any]]:
    results = []
    for index, subsampleIndices in subsampleList:
        learningResult = baseLearner.learn(sample[subsampleIndices])
        if learningResult is not None:
            if subsampleResultsDir is None:
                learningResult = baseLearner.toPickleable(learningResult)
            else:
                _SubsampleResultIO._dumpSubsampleResult(baseLearner, learningResult, subsampleResultsDir, index)
                learningResult = True
        results.append((index, learningResult))
    return results


def _subProcessObjective(subsampleResultList: List, 
                         baseLearner: BaseLearner, 
                         subsampleResultsDir: Union[str, None], 
                         sample: NDArray) -> List[NDArray]:
    objectiveList = []
    for candidate in subsampleResultList:
        if subsampleResultsDir is None:
            candidate = baseLearner.fromPickleable(candidate)
        else:
            candidate = _SubsampleResultIO._loadSubsampleResult(baseLearner, subsampleResultsDir, candidate)
        objectiveList.append(baseLearner.objective(candidate, sample))
    return objectiveList


class _CachedEvaluator:
    def __init__(self, baseLearner: BaseLearner, subsampleResultList: List, sample: NDArray, numProcesses: int, subsampleResultsDir: Union[str, None]):
        self._baseLearner: BaseLearner = baseLearner
        self._subsampleResultList: List = subsampleResultList
        self._sample: NDArray = sample
        self._numProcesses: int = max(1, int(numProcesses))
        self._subsampleResultsDir: Union[str, None] = subsampleResultsDir
        self._cachedEvaluation: List[Union[NDArray, None]] = [None] * len(sample)
        
    def _evaluateSubsamples(self, indexList: List[int], k: int, B: int, rng: np.random.Generator) -> NDArray:
        if B <= 0:
            raise ValueError(f"{self._evaluateSubsamples.__qualname__}: B = {B} <= 0")
        n = len(indexList)
        if n < k:
            raise ValueError(f"{self._evaluateSubsamples.__qualname__}: n = {n} < k = {k}")
        
        indicesToEvaluate = np.full(len(self._sample), fill_value = False)
        subsampleLists: List[List[int]] = []
        for _ in range(B):
            newSubsample = rng.choice(indexList, k, replace = False).tolist()
            subsampleLists.append(newSubsample)
            indicesToEvaluate[newSubsample] = True
        
        indicesToEvaluate: List[int] = [int(i) for i in np.flatnonzero(indicesToEvaluate) if self._cachedEvaluation[i] is None]
        indicesPerProcess: List[List[int]] = []
        processIndex = 0
        for index in indicesToEvaluate:
            if processIndex >= len(indicesPerProcess):
                indicesPerProcess.append([])
            indicesPerProcess[processIndex].append(index)
            processIndex = (processIndex + 1) % self._numProcesses
        
        if len(indicesPerProcess) <= 1:
            if len(indicesPerProcess) > 0:
                objectiveList = []
                for candidate in self._subsampleResultList:
                    if self._subsampleResultsDir is not None:
                        candidate = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, candidate)
                    objectiveList.append(self._baseLearner.objective(candidate, self._sample[indicesPerProcess[0]]))
                objectiveList = np.asarray(objectiveList, dtype = np.float64)
                if not np.isfinite(objectiveList).all():
                    raise ValueError(f"{self._evaluateSubsamples.__qualname__}: failed to evaluate all the objective values")
                for i in range(len(indicesPerProcess[0])):
                    self._cachedEvaluation[indicesPerProcess[0][i]] = objectiveList[:, i]

        else:
            if self._subsampleResultsDir is None:
                pickleableList = [self._baseLearner.toPickleable(candidate) for candidate in self._subsampleResultList]
            else:
                pickleableList = self._subsampleResultList
                
            with Pool(len(indicesPerProcess)) as pool:
                results = pool.starmap(
                    _subProcessObjective,
                    [(pickleableList, self._baseLearner, self._subsampleResultsDir, self._sample[indices]) for indices in indicesPerProcess],
                    chunksize = 1
                )

            for i, objectiveList in enumerate(results):
                objectiveList = np.asarray(objectiveList, dtype = np.float64)
                if not np.isfinite(objectiveList).all():
                    raise ValueError(f"{self._evaluateSubsamples.__qualname__}: failed to evaluate all the objective values")
                for j in range(len(indicesPerProcess[i])):
                    self._cachedEvaluation[indicesPerProcess[i][j]] = objectiveList[:, j]
        
        evalOutputList = []
        for subsample in subsampleLists:
            evalOutputList.append(np.mean([self._cachedEvaluation[index] for index in subsample], axis = 0, keepdims = False))

        return np.asarray(evalOutputList, dtype = np.float64)


class _BaseVE(metaclass = ABCMeta):
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
            subsampleLists[processIndex].append((b, self._rng.choice(n, k, replace = False).tolist()))
            processIndex = (processIndex + 1) % self._numParallelLearn

        learningResultList: List = [None] * B

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    learningResult = self._baseLearner.learn(sample[subsampleIndices])
                    if learningResult is not None and self._subsampleResultsDir is not None:
                        _SubsampleResultIO._dumpSubsampleResult(self._baseLearner, learningResult, self._subsampleResultsDir, index)
                        learningResult = index
                    learningResultList[index] = learningResult
        else:
            with Pool(len(subsampleLists)) as pool:
                results = pool.starmap(
                    _subProcessLearn, 
                    [(self._baseLearner, self._subsampleResultsDir, sample, subsampleList) for subsampleList in subsampleLists], 
                    chunksize = 1
                )
            
            for result in results:
                for index, learningResult in result:
                    if learningResult is not None:
                        if self._subsampleResultsDir is None:
                            learningResultList[index] = self._baseLearner.fromPickleable(learningResult)
                        else:
                            learningResultList[index] = index

        return [entry for entry in learningResultList if entry is not None]
        
    @abstractmethod
    def run(self, sample: NDArray) -> Any:
        pass


class MoVE(_BaseVE):
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
        if self._subsampleResultsDir is not None:
            _SubsampleResultIO._prepareSubsampleResultDir(self._subsampleResultsDir)

        learningResults = self._learnOnSubsamples(np.asarray(sample), k, B)

        if len(learningResults) == 0:
            raise ValueError(f"{self.run.__qualname__}: empty candidate set")
        
        indexCountPairs: List[List[int]] = []
        maxIndex = 0
        maxCount = 0
        for i in range(len(learningResults)):
            result1 = learningResults[i]
            if self._subsampleResultsDir is not None:
                result1 = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, result1)
            index = len(indexCountPairs)
            for j in range(len(indexCountPairs)):
                result2 = learningResults[indexCountPairs[j][0]]
                if self._subsampleResultsDir is not None:
                    result2 = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, result2)
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
            output = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, learningResults[maxIndex])
            if self._deleteSubsampleResults:
                _SubsampleResultIO._deleteSubsampleResults(self._subsampleResultsDir, learningResults)
            return output


class ROVE(_BaseVE):
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
                Whether or not (ROVEs vs ROVE) to split the data across the model candidate retrieval phase and the voting phase.
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
                Subsample size for the voting phase.
            B1: 
                Number of subsamples to draw in the model candidate retrieval phase.
            B2: 
                Number of subsamples to draw in the voting phase.
            epsilon: 
                The suboptimality threshold. Any value < 0 leads to auto-selection. Default -1.0.
            autoEpsilonProb: 
                The probability threshold guiding the auto-selection of epsilon. Default 0.5.

        Returns:
            A learning result.
        """
        sample = np.asarray(sample)
        n1 = len(sample)
        n2 = 0

        if self._dataSplit:
            n1 = len(sample) // 2
            if n1 <= 0 or n1 >= len(sample):
                raise ValueError(f"{self.run.__qualname__}: insufficient data n = {len(sample)}")
            n2 = n1

        if self._subsampleResultsDir is not None:
            _SubsampleResultIO._prepareSubsampleResultDir(self._subsampleResultsDir)

        learningResults = self._learnOnSubsamples(sample[:n1], k1, B1)

        retrievedList: List = []
        if self._baseLearner.enableDeduplication:
            for i in range(len(learningResults)):
                result1 = learningResults[i]
                if self._subsampleResultsDir is not None:
                    result1 = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, result1)
                existing = False
                for result2 in retrievedList:
                    if self._subsampleResultsDir is not None:
                        result2 = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, result2)
                    if self._baseLearner.isDuplicate(result1, result2):
                        existing = True
                        break

                if not existing:
                    retrievedList.append(learningResults[i])
        else:
            retrievedList = learningResults

        if len(retrievedList) == 0:
            raise ValueError(f"{self.run.__qualname__}: failed to retrieve any learning result")
        
        cachedEvaluator = _CachedEvaluator(self._baseLearner, retrievedList, sample, self._numParallelEval, self._subsampleResultsDir)
        
        evalArray = cachedEvaluator._evaluateSubsamples([i for i in range(n2, len(sample))], k2, B2, self._rng)
        gapMatrix = self._gapMatrix(evalArray)

        if epsilon < 0:
            autoEpsilonProb = min(max(0, autoEpsilonProb), 1)
            if self._dataSplit:
                evalArray = cachedEvaluator._evaluateSubsamples([i for i in range(0, n1)], k2, B2, self._rng)
                gapMatrix1 = self._gapMatrix(evalArray)
                epsilon = ROVE._findEpsilon(gapMatrix1, autoEpsilonProb)
            else:
                epsilon = ROVE._findEpsilon(gapMatrix, autoEpsilonProb)
    
        probArray = ROVE._epsilonOptimalProb(gapMatrix, epsilon)

        if self._subsampleResultsDir is None:
            return retrievedList[np.argmax(probArray)]
        else:
            output = _SubsampleResultIO._loadSubsampleResult(self._baseLearner, self._subsampleResultsDir, retrievedList[np.argmax(probArray)])
            if self._deleteSubsampleResults:
                _SubsampleResultIO._deleteSubsampleResults(self._subsampleResultsDir, learningResults)
            return output