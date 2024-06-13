from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Process, Queue
from typing import List, Any, Tuple, Dict


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
        pass

    @abstractmethod
    def serialize(self, trainingOutput: Any) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, serializedTrainingOutput: bytes) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def identicalTrainingOuputs(output1: Any, output2: Any) -> bool:
        pass

    def _subProcessTrain(self, sample: NDArray, subsampleList: List[Tuple[int, List[int]]], queue: Queue):
        for index, subsampleIndices in subsampleList:
            trainingOutput = self.train(sample[subsampleIndices])
            if trainingOutput is None:
                queue.put((index, trainingOutput))
            else:
                queue.put((index, self.serialize(trainingOutput)))

    def _majorityVote(self, sample: NDArray, B: int, k: int):
        if B <= 0:
            raise ValueError(f"B = {B} <= 0")
        sample = np.asarray(sample)
        n = sample.shape[0]
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

        candidateList: List = [None for _ in range(B)]

        if len(subsampleLists) <= 1:
            for subsampleList in subsampleLists:
                for index, subsampleIndices in subsampleList:
                    candidateList[index] = self.train(sample[subsampleIndices])
        else:
            queue = Queue()
            processList: List[Process] = [Process(target = self._subProcessTrain, args = (sample, subsampleList, queue), daemon = True) for subsampleList in subsampleLists]
            
            for process in processList:
                process.start()

            for _ in range(B):
                index, trainingOutput = queue.get()
                if trainingOutput is not None:
                    candidateList[index] = self.deserialize(trainingOutput)

            for process in processList:
                process.join()

        candidateCount: List[int] = [0 for _ in range(B)]
        maxIndex = -1
        maxCount = -1
        for i in range(len(candidateList)):
            if candidateList[i] is None:
                continue

            index = i
            for j in range(i):
                if candidateCount[j] > 0:
                    if self.identicalTrainingOuputs(candidateList[i], candidateList[j]):
                        index = j
                        break
            
            candidateCount[index] += 1
            if candidateCount[index] > maxCount:
                maxIndex = index
                maxCount = candidateCount[index]
        
        if maxIndex >= 0:
            return candidateList[maxIndex]
        else:
            raise RuntimeError("all subsampled training tasks have failed")

class ReBAG(metaclass = ABCMeta):
    pass