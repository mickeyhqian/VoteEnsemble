from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Queue
from typing import List, Any, Tuple



class BAG(metaclass = ABCMeta):
    def __init__(self, numParallel: int = 1, randomState: np.random.Generator | int | None = None):
        if isinstance(randomState, np.random.Generator):
            self._rng = randomState
        elif isinstance(randomState, int) and randomState >= 0:
            self._rng = np.random.default_rng(seed = randomState)
        else:
            self._rng = np.random.default_rng()
        self._numParallel: int = max(1, numParallel)

    @abstractmethod
    def train(self, sample: NDArray) -> Any:
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

    def serialize(self, trainingOutput: Any) -> bytes:
        """
        serialization method for training results
        
        To be overridden if the output of self.train may not be pickleable and parallel training is enabled (self._numParallel > 1)
        """
        return trainingOutput

    def deserialize(self, serializedTrainingOutput: bytes) -> Any:
        """
        deserialization method for training results

        To be overridden if the output of self.train may not be pickleable and parallel training is enabled (self._numParallel > 1)
        """
        return serializedTrainingOutput

    def _subProcessTrain(self, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        for index, subsampleIndices in subsampleList:
            trainingOutput = self.train(sample[subsampleIndices])
            if trainingOutput is None:
                queue.put((index, trainingOutput))
            else:
                queue.put((index, self.serialize(trainingOutput)))

    def _trainOnSubsamples(self, sample: NDArray, k: int, B: int) -> List:
        if B <= 0:
            raise ValueError(f"B = {B} <= 0")
        n: int = len(sample)
        if n < k:
            raise ValueError(f"n = {n} < k = {k}")
        
        subsampleLists: List[List[Tuple[int, List[int]]]] = []
        processIndex = 0
        for b in range(B):
            if processIndex >= len(subsampleLists):
                subsampleLists.append([])
            newSubsample = self._rng.choice(n, k, replace=False)
            subsampleLists[processIndex].append((b, newSubsample.tolist()))
            processIndex = (processIndex + 1) % self._numParallel

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
                    trainingOutputList[index] = self.deserialize(trainingOutput)

            for process in processList:
                process.join()

        return [entry for entry in trainingOutputList if entry is not None]

    def _majorityVote(self, trainingOutputList: List) -> Any:
        if len(trainingOutputList) == 0:
            return None
        
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
    def __init__(self, dataSplit: bool, numParallel: int = 1, randomState: np.random.Generator | int | None = None):
        super().__init__(numParallel = numParallel, randomState = randomState)
        self._dataSplit: bool = dataSplit

    @abstractmethod
    def trainingObjective(self, trainingOutput: Any, sample: NDArray) -> float:
        """
        compute the training objective for a training result on a data set (must be consistent with the training objective minimized by self.train)

        trainingOutput: a training result, e.g., a solution vector (for optimization problems) or a machine learning model (for machine learning problems)
        sample: numpy array of training data, where each sample[i] for i in range(len(sample)) is a data point

        return the training objective value
        """
        pass

    def run(self, sample: NDArray, k1: int, k2: int, B1: int, B2: int) -> Any:
        sample = np.asarray(sample)
        n: int = len(sample)
        if n < 2:
            return None
        
        sample1 = sample
        sample2 = sample
        if self._dataSplit:
            sample1 = sample[:n//2]
            sample2 = sample[n//2:]

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
            return None
        
        