import signal
from Bagging import BaseTrainer, BAG, ReBAG
import numpy as np
from numpy.typing import NDArray
import pickle
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Callable, Any
import logging
logger = logging.getLogger(name = "Bagging")



class GracefulKiller:
    _killed: bool = False

    @classmethod
    def kill(cls, *args):
        cls._killed = True

    @classmethod
    def isKilled(cls) -> bool:
        return cls._killed

    @classmethod
    def setup(cls):
        signal.signal(signal.SIGINT, cls.kill)
        signal.signal(signal.SIGTERM, cls.kill)


class KilledByUser(Exception):
    pass

def checkKiller():
    if GracefulKiller.isKilled():
        raise KilledByUser()


def runTraining(baseTrainer: BaseTrainer, 
                sampler: Callable[[int, np.random.Generator], NDArray],
                sampleSizeList: List[int], 
                kList: List[Tuple[int, float]], 
                BList: List[int], 
                k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                B12List: List[Tuple[int, int]], 
                numReplicates: int,
                numParallelTrain: int = 1,
                numParallelEval: int = 1):
    baseList = [[] for _ in range(len(sampleSizeList))]
    BAGList = [[[[] for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    ReBAGList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ReBAGSList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]

    for i in range(len(sampleSizeList)):
        n = sampleSizeList[i]
        sampleRNG = np.random.default_rng(seed = 888)
        for j in range(numReplicates):
            sample = sampler(n, sampleRNG)
            
            checkKiller()
            baseResult = baseTrainer.train(sample)
            if baseResult is None:
                raise RuntimeError(f"base training failed for sample size {n}, replication {j}")
            baseList[i].append(baseTrainer.toPickleable(baseResult))
            logger.info(f"Finish base training for sample size {n}, replication {j}")

            for ind1, B in enumerate(BList):
                for ind2, (base, ratio) in enumerate(kList):
                    k = max(base, int(n * ratio))
                    bag = BAG(baseTrainer, numParallelTrain = numParallelTrain, randomState = 666)

                    checkKiller()
                    BAGList[i][ind1][ind2].append(baseTrainer.toPickleable(bag.run(sample, k, B)))
                    logger.info(f"Finish BAG training for sample size {n}, replication {j}, B={B}, k={k}")

            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n * ratio1))
                    k2 = max(base2, int(n * ratio2))
                    rebag = ReBAG(baseTrainer, False, numParallelEval = numParallelEval, numParallelTrain = numParallelTrain, randomState = 666)
                    
                    checkKiller()
                    ReBAGList[i][ind1][ind2].append(baseTrainer.toPickleable(rebag.run(sample, k1, k2, B1, B2)))
                    logger.info(f"Finish ReBAG training for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2 = {k2}")

            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n / 2 * ratio1))
                    k2 = max(base2, int(n / 2 * ratio2))
                    rebags = ReBAG(baseTrainer, True, numParallelEval = numParallelEval, numParallelTrain = numParallelTrain, randomState = 666)

                    checkKiller()
                    ReBAGSList[i][ind1][ind2].append(baseTrainer.toPickleable(rebags.run(sample, k1, k2, B1, B2)))
                    logger.info(f"Finish ReBAG-S training for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2 = {k2}")

    return baseList, BAGList, ReBAGList, ReBAGSList


def runEvaluation(baseTrainer: BaseTrainer, 
                  baseList: List, 
                  BAGList: List, 
                  ReBAGList: List, 
                  ReBAGSList: List, 
                  evaluator: Callable[[Any], float],
                  sampleSizeList: List[int], 
                  kList: List[Tuple[int, float]], 
                  BList: List[int], 
                  k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                  B12List: List[Tuple[int, int]],
                  numReplicates: int):

    if numReplicates <= 0:
        raise ValueError("numReplicates must be >= 1")
    
    baseObjList = [[] for _ in range(len(sampleSizeList))]
    baseObjAvg = [None for _ in range(len(sampleSizeList))]
    BAGObjList = [[[[] for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    BAGObjAvg = [[[None for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    ReBAGObjList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ReBAGObjAvg = [[[None for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ReBAGSObjList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ReBAGSObjAvg = [[[None for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]

    for i in range(len(sampleSizeList)):
        for j in range(numReplicates):
            checkKiller()
            baseObjList[i].append(evaluator(baseTrainer.fromPickleable(baseList[i][j])))
            logger.info(f"Finish base evaluation for sample size {sampleSizeList[i]}, replication {j}")

            for ind1 in range(len(BList)):
                for ind2 in range(len(kList)):
                    checkKiller()
                    BAGObjList[i][ind1][ind2].append(evaluator(baseTrainer.fromPickleable(BAGList[i][ind1][ind2][j])))
                    logger.info(f"Finish BAG evaluation for sample size {sampleSizeList[i]}, replication {j}, B={BList[ind1]}, k={kList[ind2]}")

            for ind1 in range(len(B12List)):
                for ind2 in range(len(k12List)):
                    checkKiller()
                    ReBAGObjList[i][ind1][ind2].append(evaluator(baseTrainer.fromPickleable(ReBAGList[i][ind1][ind2][j])))
                    logger.info(f"Finish ReBAG evaluation for sample size {sampleSizeList[i]}, replication {j}, B12={B12List[ind1]}, k12 = {k12List[ind2]}")
                    checkKiller()
                    ReBAGSObjList[i][ind1][ind2].append(evaluator(baseTrainer.fromPickleable(ReBAGSList[i][ind1][ind2][j])))
                    logger.info(f"Finish ReBAG-S evaluation for sample size {sampleSizeList[i]}, replication {j}, B12={B12List[ind1]}, k12 = {k12List[ind2]}")
    
        baseObjAvg[i] = np.mean(baseObjList[i])
        for ind1 in range(len(BList)):
            for ind2 in range(len(kList)):
                BAGObjAvg[i][ind1][ind2] = np.mean(BAGObjList[i][ind1][ind2])
        for ind1 in range(len(B12List)):
            for ind2 in range(len(k12List)):
                ReBAGObjAvg[i][ind1][ind2] = np.mean(ReBAGObjList[i][ind1][ind2])
                ReBAGSObjAvg[i][ind1][ind2] = np.mean(ReBAGSObjList[i][ind1][ind2])

    return baseObjList, BAGObjList, ReBAGObjList, ReBAGSObjList, baseObjAvg, BAGObjAvg, ReBAGObjAvg, ReBAGSObjAvg


def dumpTrainingResults(baseList: List, 
                        BAGList: List, 
                        ReBAGList: List, 
                        ReBAGSList: List,
                        sampleSizeList: List[int], 
                        kList: List[Tuple[int, float]], 
                        BList: List[int], 
                        k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                        B12List: List[Tuple[int, int]], 
                        numReplicates: int,
                        filePath: str):
    os.makedirs(os.path.dirname(filePath), exist_ok = True)
    with open(filePath, "wb") as f:
        pickle.dump((
            baseList,
            BAGList, 
            ReBAGList, 
            ReBAGSList,
            sampleSizeList, 
            kList, 
            BList, 
            k12List, 
            B12List, 
            numReplicates,
        ), f)
    logger.info(f"dumped training results to {filePath}")


def dumpEvalResults(baseObjList: List, 
                    BAGObjList: List, 
                    ReBAGObjList: List, 
                    ReBAGSObjList: List, 
                    baseObjAvg: List, 
                    BAGObjAvg: List, 
                    ReBAGObjAvg: List, 
                    ReBAGSObjAvg: List,
                    sampleSizeList: List[int], 
                    kList: List[Tuple[int, float]], 
                    BList: List[int], 
                    k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                    B12List: List[Tuple[int, int]],
                    numReplicates: int,
                    filePath: str):
    os.makedirs(os.path.dirname(filePath), exist_ok = True)
    with open(filePath, "wb") as f:
        pickle.dump((
            baseObjList, 
            BAGObjList, 
            ReBAGObjList, 
            ReBAGSObjList, 
            baseObjAvg, 
            BAGObjAvg, 
            ReBAGObjAvg, 
            ReBAGSObjAvg,
            sampleSizeList, 
            kList, 
            BList, 
            k12List, 
            B12List, 
            numReplicates,
        ), f)
    logger.info(f"dumped evaluation results to {filePath}")


def loadResults(filePath: str):
    with open(filePath, "rb") as f:
        return pickle.load(f)


def plotAverage(baseObjAvg: List, 
                BAGObjAvg: List, 
                ReBAGObjAvg: List, 
                ReBAGSObjAvg: List, 
                sampleSizeList: List, 
                kList: List, 
                BList: List, 
                k12List: List, 
                B12List: List,
                filePath: str,
                xLogScale: bool = True,
                yLogScale: bool = False):
    fig, ax = plt.subplots()
    ax.plot(sampleSizeList, baseObjAvg, marker = 'o', markeredgecolor = 'none', color = 'blue', linestyle = 'solid', label = 'base')

    for ind1, B in enumerate(BList):
        for ind2, k in enumerate(kList):
            ax.plot(sampleSizeList, [BAGObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'BAG, B={B}, k={k}')
    
    for ind1, B12 in enumerate(B12List):
        for ind2, k in enumerate(k12List):
            ax.plot(sampleSizeList, [ReBAGObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'ReBAG, B12={B12}, k={k}')
            ax.plot(sampleSizeList, [ReBAGSObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'ReBAG-S, B12={B12}, k={k}')
    
    ax.set_xlabel('sample size', size = 20)
    ax.set_ylabel('cost', size = 20)
    if xLogScale:
        ax.set_xscale('log')
    if yLogScale:
        ax.set_yscale('log')
    ax.legend(fontsize = 'small')

    os.makedirs(os.path.dirname(filePath), exist_ok = True)
    fig.savefig(filePath, dpi = 600)
    logger.info(f"saved a plot of average performance to {filePath}")


def plotCDF(baseObjList: List, 
            BAGObjList: List, 
            ReBAGObjList: List, 
            ReBAGSObjList: List, 
            sampleSizeList: List, 
            kList: List, 
            BList: List, 
            k12List: List, 
            B12List: List,
            filePath: str,
            xLogScale: bool = False,
            yLogScale: bool = True):
    fig, ax = plt.subplots(nrows = len(sampleSizeList), figsize = (6, len(sampleSizeList) * 4))
    if len(sampleSizeList) <= 1:
        ax = [ax]

    def getCDF(sequence):
        xList = []
        yList = []
        for num in sorted(sequence):
            if len(xList) == 0:
                xList.append(num)
                yList.append(1 / len(sequence))
            elif num > xList[-1]:
                xList.append(num)
                yList.append(yList[-1] + 1 / len(sequence))
            else:
                yList[-1] += 1 / len(sequence)
        
        tailList = []
        for i in range(len(yList)):
            if i == 0:
                tailList.append(1)
            else:
                tailList.append(1 - yList[i - 1])

        return xList, tailList

    for i in range(len(sampleSizeList)):
        xList, yList = getCDF(baseObjList[i])
        ax[i].plot(xList, yList, marker = 'o', markeredgecolor = 'none', color = 'blue', linestyle = 'solid', label = 'base')
        for ind1, B in enumerate(BList):
            for ind2, k in enumerate(kList):
                xList, yList = getCDF(BAGObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'BAG, B={B}, k={k}')
        
        for ind1, B12 in enumerate(B12List):
            for ind2, k in enumerate(k12List):
                xList, yList = getCDF(ReBAGObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'ReBAG, B12={B12}, k={k}')
                xList, yList = getCDF(ReBAGSObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'ReBAG-S, B12={B12}, k={k}')
        
        if i == len(sampleSizeList) - 1:
            ax[i].set_xlabel('cost')

        ax[i].set_ylabel('tail prob')
        if xLogScale:
            ax[i].set_xscale('log')
        if yLogScale:
            ax[i].set_yscale('log')
        ax[i].set_title(f'sample size = {sampleSizeList[i]}', fontsize = "medium")

    # Create a legend using the first subplot
    handles, labels = ax[0].get_legend_handles_labels()

    # Place the combined legend outside the subplots
    fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.95), fontsize = 'small')

    os.makedirs(os.path.dirname(filePath), exist_ok = True)
    fig.savefig(filePath, dpi = 600)
    logger.info(f"saved plots of performance CDFs to {filePath}")


def pipeline(resultDir: str,
             baseTrainer: BaseTrainer, 
             sampler: Callable[[int, np.random.Generator], NDArray],
             evaluator: Callable[[Any], float],
             sampleSizeList: List, 
             kList: List, 
             BList: List, 
             k12List: List, 
             B12List: List, 
             numReplicates: int,
             numParallelTrain: int = 1,
             numParallelEval: int = 1):

    baseList = [] 
    BAGList = []
    ReBAGList = []
    ReBAGSList = []
    trainingResultFile = os.path.join(resultDir, "trainingResults.pkl")

    baseObjList = []
    BAGObjList = []
    ReBAGObjList = []
    ReBAGSObjList = []
    baseObjAvg = []
    BAGObjAvg = []
    ReBAGObjAvg = []
    ReBAGSObjAvg = []
    evalResultFile = os.path.join(resultDir, "evalResults.pkl")

    sampleSizeFinished = []

    avgFigPath = os.path.join(resultDir, "avgPlot.png")
    cdfFigPath = os.path.join(resultDir, "cdfPlot.png")

    if os.path.isfile(trainingResultFile):
        trainingResults = loadResults(trainingResultFile)
        if len(trainingResults) == 11:
            trainingResults = trainingResults[1:]
        (
            baseList, 
            BAGList, 
            ReBAGList, 
            ReBAGSList,
            sampleSizeFinished, 
            _, 
            _, 
            _, 
            _, 
            _,
        ) = trainingResults
        logger.info("Loaded existing training results")

    for i in range(len(sampleSizeFinished)):
        if i >= len(sampleSizeList) or sampleSizeFinished[i] != sampleSizeList[i]:
            raise ValueError("incorrect existing training results")

    GracefulKiller.setup()

    try:
        i = 0
        while i < len(sampleSizeList):
            checkKiller()

            if i < len(sampleSizeFinished):
                baseListNew, BAGListNew, ReBAGListNew, ReBAGSListNew = baseList[:len(sampleSizeFinished)], BAGList[:len(sampleSizeFinished)], ReBAGList[:len(sampleSizeFinished)], ReBAGSList[:len(sampleSizeFinished)]
                evalSampleSizeList = sampleSizeFinished
                i = len(sampleSizeFinished)
            else: 
                baseListNew, BAGListNew, ReBAGListNew, ReBAGSListNew = runTraining(baseTrainer, 
                                                                                   sampler,
                                                                                   [sampleSizeList[i]], 
                                                                                   kList, 
                                                                                   BList, 
                                                                                   k12List, 
                                                                                   B12List, 
                                                                                   numReplicates,
                                                                                   numParallelTrain = numParallelTrain,
                                                                                   numParallelEval = numParallelEval)
                evalSampleSizeList = [sampleSizeList[i]]
                baseList.extend(baseListNew)
                BAGList.extend(BAGListNew)
                ReBAGList.extend(ReBAGListNew)
                ReBAGSList.extend(ReBAGSListNew)
                sampleSizeFinished.extend(evalSampleSizeList)
                i += 1

                dumpTrainingResults(baseList, 
                                    BAGList, 
                                    ReBAGList, 
                                    ReBAGSList,
                                    sampleSizeFinished, 
                                    kList, 
                                    BList, 
                                    k12List, 
                                    B12List, 
                                    numReplicates,
                                    trainingResultFile)
            

            baseObjListNew, BAGObjListNew, ReBAGObjListNew, ReBAGSObjListNew, baseObjAvgNew, BAGObjAvgNew, ReBAGObjAvgNew, ReBAGSObjAvgNew = runEvaluation(baseTrainer,
                                                                                                                                                           baseListNew, 
                                                                                                                                                           BAGListNew, 
                                                                                                                                                           ReBAGListNew, 
                                                                                                                                                           ReBAGSListNew, 
                                                                                                                                                           evaluator,
                                                                                                                                                           evalSampleSizeList, 
                                                                                                                                                           kList, 
                                                                                                                                                           BList, 
                                                                                                                                                           k12List, 
                                                                                                                                                           B12List,
                                                                                                                                                           numReplicates)
            
            baseObjList.extend(baseObjListNew)
            BAGObjList.extend(BAGObjListNew)
            ReBAGObjList.extend(ReBAGObjListNew)
            ReBAGSObjList.extend(ReBAGSObjListNew)
            baseObjAvg.extend(baseObjAvgNew)
            BAGObjAvg.extend(BAGObjAvgNew)
            ReBAGObjAvg.extend(ReBAGObjAvgNew)
            ReBAGSObjAvg.extend(ReBAGSObjAvgNew)
            
            dumpEvalResults(baseObjList, 
                            BAGObjList, 
                            ReBAGObjList, 
                            ReBAGSObjList, 
                            baseObjAvg, 
                            BAGObjAvg, 
                            ReBAGObjAvg, 
                            ReBAGSObjAvg,
                            sampleSizeFinished, 
                            kList, 
                            BList, 
                            k12List, 
                            B12List,
                            numReplicates,
                            evalResultFile)
            
            plotAverage(baseObjAvg, 
                        BAGObjAvg, 
                        ReBAGObjAvg, 
                        ReBAGSObjAvg,
                        sampleSizeFinished, 
                        kList, 
                        BList, 
                        k12List, 
                        B12List,
                        avgFigPath)
            
            plotCDF(baseObjList, 
                    BAGObjList, 
                    ReBAGObjList, 
                    ReBAGSObjList, 
                    sampleSizeFinished, 
                    kList, 
                    BList, 
                    k12List, 
                    B12List,
                    cdfFigPath)
            
    except KilledByUser:
        logger.info("The experiment is killed")