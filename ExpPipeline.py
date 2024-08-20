import signal
from VoteEnsemble import BaseLearner, MoVE, ROVE
import numpy as np
from numpy.typing import NDArray
import pickle
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Callable, Any, Union
import logging
logger = logging.getLogger(name = "VE")



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


def runTraining(baseLearner: BaseLearner, 
                sampler: Callable[[int, int, np.random.Generator], NDArray],
                sampleSizeList: List[int], 
                kList: List[Tuple[int, float]], 
                BList: List[int], 
                k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                B12List: List[Tuple[int, int]], 
                numReplicates: int,
                resultDir: str,
                numParallelLearn: int = 1,
                numParallelEval: int = 1,
                subsampleResultsDir: Union[str, None] = None,
                runConventionalBagging: bool = False):
    baseList = [[] for _ in range(len(sampleSizeList))]
    MoVEList = [[[[] for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    ROVEList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ROVEsList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    os.makedirs(resultDir, exist_ok = True)

    for i in range(len(sampleSizeList)):
        n = sampleSizeList[i]
        sampleRNG = np.random.default_rng(seed = 888)
        for j in range(numReplicates):
            sample = sampler(n, j, sampleRNG)
            
            checkKiller()
            resultFile = os.path.join(resultDir, f"base_{n}_{j}")
            if not os.path.isfile(resultFile):
                learningResult = baseLearner.learn(sample)
                if learningResult is None:
                    raise RuntimeError(f"base learning failed for sample size {n}, replication {j}")
                baseLearner.dumpLearningResult(learningResult, resultFile)
            baseList[i].append(resultFile)
            logger.info(f"Finish base learning for sample size {n}, replication {j}")

            for ind1, B in enumerate(BList):
                for ind2, (base, ratio) in enumerate(kList):
                    k = max(base, int(n * ratio))
                    move = MoVE(baseLearner, numParallelLearn = numParallelLearn, randomState = 666, subsampleResultsDir = subsampleResultsDir)

                    checkKiller()
                    resultFile = os.path.join(resultDir, f"{MoVE.__name__}_{n}_{j}_{B}_{k}")
                    if not os.path.isfile(resultFile):
                        baseLearner.dumpLearningResult(move.run(sample, k, B), resultFile)
                    MoVEList[i][ind1][ind2].append(resultFile)
                    logger.info(f"Finish {MoVE.__name__} learning for sample size {n}, replication {j}, B={B}, k={k}")

            caseSet = set()
            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n * ratio1))
                    k2 = max(base2, int(n * ratio2))
                    deleteSubsampleResults = not runConventionalBagging or (B1, k1) in caseSet
                    if deleteSubsampleResults:
                        ROVESubsampleResultsDir = subsampleResultsDir
                    else:
                        caseSet.add((B1, k1))
                        ROVESubsampleResultsDir = os.path.join(subsampleResultsDir, f"{ROVE.__name__}Results_{n}_{j}_{B1}_{k1}")
                    rove = ROVE(baseLearner, False, numParallelEval = numParallelEval, numParallelLearn = numParallelLearn, randomState = 666, subsampleResultsDir = ROVESubsampleResultsDir, deleteSubsampleResults = deleteSubsampleResults)
                    
                    checkKiller()
                    resultFile = os.path.join(resultDir, f"{ROVE.__name__}_{n}_{j}_{B1}_{B2}_{k1}_{k2}")
                    if not os.path.isfile(resultFile):
                        baseLearner.dumpLearningResult(rove.run(sample, k1, k2, B1, B2), resultFile)
                    ROVEList[i][ind1][ind2].append(resultFile)
                    logger.info(f"Finish {ROVE.__name__} learning for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2={k2}")

            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n / 2 * ratio1))
                    k2 = max(base2, int(n / 2 * ratio2))
                    roves = ROVE(baseLearner, True, numParallelEval = numParallelEval, numParallelLearn = numParallelLearn, randomState = 666, subsampleResultsDir = subsampleResultsDir)

                    checkKiller()
                    resultFile = os.path.join(resultDir, f"{ROVE.__name__}s_{n}_{j}_{B1}_{B2}_{k1}_{k2}")
                    if not os.path.isfile(resultFile):
                        baseLearner.dumpLearningResult(roves.run(sample, k1, k2, B1, B2), resultFile)
                    ROVEsList[i][ind1][ind2].append(resultFile)
                    logger.info(f"Finish {ROVE.__name__}s learning for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2={k2}")

    return baseList, MoVEList, ROVEList, ROVEsList


def runEvaluation(baseLearner: BaseLearner, 
                  baseList: List, 
                  MoVEList: List, 
                  ROVEList: List, 
                  ROVEsList: List, 
                  evaluator: Callable[[Any, int], float],
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
    MoVEObjList = [[[[] for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    MoVEObjAvg = [[[None for _ in range(len(kList))] for _ in range(len(BList))] for _ in range(len(sampleSizeList))]
    ROVEObjList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ROVEObjAvg = [[[None for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ROVEsObjList = [[[[] for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]
    ROVEsObjAvg = [[[None for _ in range(len(k12List))] for _ in range(len(B12List))] for _ in range(len(sampleSizeList))]

    for i in range(len(sampleSizeList)):
        for j in range(numReplicates):
            checkKiller()
            baseObjList[i].append(evaluator(baseLearner.loadLearningResult(baseList[i][j]), j))
            logger.info(f"Finish base evaluation for sample size {sampleSizeList[i]}, replication {j}")

            for ind1 in range(len(BList)):
                for ind2 in range(len(kList)):
                    checkKiller()
                    MoVEObjList[i][ind1][ind2].append(evaluator(baseLearner.loadLearningResult(MoVEList[i][ind1][ind2][j]), j))
                    logger.info(f"Finish {MoVE.__name__} evaluation for sample size {sampleSizeList[i]}, replication {j}, B={BList[ind1]}, k={kList[ind2]}")

            for ind1 in range(len(B12List)):
                for ind2 in range(len(k12List)):
                    checkKiller()
                    ROVEObjList[i][ind1][ind2].append(evaluator(baseLearner.loadLearningResult(ROVEList[i][ind1][ind2][j]), j))
                    logger.info(f"Finish {ROVE.__name__} evaluation for sample size {sampleSizeList[i]}, replication {j}, B12={B12List[ind1]}, k12={k12List[ind2]}")
                    checkKiller()
                    ROVEsObjList[i][ind1][ind2].append(evaluator(baseLearner.loadLearningResult(ROVEsList[i][ind1][ind2][j]), j))
                    logger.info(f"Finish {ROVE.__name__}s evaluation for sample size {sampleSizeList[i]}, replication {j}, B12={B12List[ind1]}, k12={k12List[ind2]}")
    
        baseObjAvg[i] = np.mean(baseObjList[i])
        for ind1 in range(len(BList)):
            for ind2 in range(len(kList)):
                MoVEObjAvg[i][ind1][ind2] = np.mean(MoVEObjList[i][ind1][ind2])
        for ind1 in range(len(B12List)):
            for ind2 in range(len(k12List)):
                ROVEObjAvg[i][ind1][ind2] = np.mean(ROVEObjList[i][ind1][ind2])
                ROVEsObjAvg[i][ind1][ind2] = np.mean(ROVEsObjList[i][ind1][ind2])

    return baseObjList, MoVEObjList, ROVEObjList, ROVEsObjList, baseObjAvg, MoVEObjAvg, ROVEObjAvg, ROVEsObjAvg


def dumpEvalResults(baseObjList: List, 
                    MoVEObjList: List, 
                    ROVEObjList: List, 
                    ROVEsObjList: List, 
                    baseObjAvg: List, 
                    MoVEObjAvg: List, 
                    ROVEObjAvg: List, 
                    ROVEsObjAvg: List,
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
            MoVEObjList, 
            ROVEObjList, 
            ROVEsObjList, 
            baseObjAvg, 
            MoVEObjAvg, 
            ROVEObjAvg, 
            ROVEsObjAvg,
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
                MoVEObjAvg: List, 
                ROVEObjAvg: List, 
                ROVEsObjAvg: List, 
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
            ax.plot(sampleSizeList, [MoVEObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'{MoVE.__name__}, B={B}, k={k}')
    
    for ind1, B12 in enumerate(B12List):
        for ind2, k in enumerate(k12List):
            ax.plot(sampleSizeList, [ROVEObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'{ROVE.__name__}, B12={B12}, k12={k}')
            ax.plot(sampleSizeList, [ROVEsObjAvg[i][ind1][ind2] for i in range(len(sampleSizeList))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', label = f'{ROVE.__name__}s, B12={B12}, k12={k}')
    
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
            MoVEObjList: List, 
            ROVEObjList: List, 
            ROVEsObjList: List, 
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
        ax[i].plot(xList, yList, color = 'blue', linestyle = 'solid', label = 'base', linewidth = 2)
        for ind1, B in enumerate(BList):
            for ind2, k in enumerate(kList):
                xList, yList = getCDF(MoVEObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, linestyle = 'solid', label = f'{MoVE.__name__}, B={B}, k={k}', linewidth = 2)
        
        for ind1, B12 in enumerate(B12List):
            for ind2, k in enumerate(k12List):
                xList, yList = getCDF(ROVEObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, linestyle = 'solid', label = f'{ROVE.__name__}, B12={B12}, k12={k}', linewidth = 2)
                xList, yList = getCDF(ROVEsObjList[i][ind1][ind2])
                ax[i].plot(xList, yList, linestyle = 'solid', label = f'{ROVE.__name__}s, B12={B12}, k12={k}', linewidth = 2)
        
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
             baseLearner: BaseLearner, 
             sampler: Callable[[int, int, np.random.Generator], NDArray],
             evaluator: Callable[[Any, int], float],
             sampleSizeList: List, 
             kList: List, 
             BList: List, 
             k12List: List, 
             B12List: List, 
             numReplicates: int,
             numParallelLearn: int = 1,
             numParallelEval: int = 1,
             dumpSubsampleResults: bool = False,
             runConventionalBagging: bool = False):

    baseList = [] 
    MoVEList = []
    ROVEList = []
    ROVEsList = []
    learningResultDir = os.path.join(resultDir, "learningResults")
    subsampleResultsDir = None
    if dumpSubsampleResults:
        subsampleResultsDir = os.path.join(resultDir, "subsampleResults")

    baseObjList = []
    MoVEObjList = []
    ROVEObjList = []
    ROVEsObjList = []
    baseObjAvg = []
    MoVEObjAvg = []
    ROVEObjAvg = []
    ROVEsObjAvg = []
    evalResultFile = os.path.join(resultDir, "evalResults.pkl")

    sampleSizeFinished = []

    avgFigPath = os.path.join(resultDir, "avgPlot.png")
    cdfFigPath = os.path.join(resultDir, "cdfPlot.png")

    GracefulKiller.setup()

    try:
        for sampleSize in sampleSizeList:
            checkKiller()

            baseListNew, MoVEListNew, ROVEListNew, ROVEsListNew = runTraining(baseLearner, 
                                                                              sampler,
                                                                              [sampleSize], 
                                                                              kList, 
                                                                              BList, 
                                                                              k12List, 
                                                                              B12List, 
                                                                              numReplicates,
                                                                              learningResultDir,
                                                                              numParallelLearn = numParallelLearn,
                                                                              numParallelEval = numParallelEval,
                                                                              subsampleResultsDir = subsampleResultsDir,
                                                                              runConventionalBagging = runConventionalBagging)

            baseList.extend(baseListNew)
            MoVEList.extend(MoVEListNew)
            ROVEList.extend(ROVEListNew)
            ROVEsList.extend(ROVEsListNew)

            baseObjListNew, MoVEObjListNew, ROVEObjListNew, ROVEsObjListNew, baseObjAvgNew, MoVEObjAvgNew, ROVEObjAvgNew, ROVEsObjAvgNew = runEvaluation(baseLearner,
                                                                                                                                                         baseListNew, 
                                                                                                                                                         MoVEListNew, 
                                                                                                                                                         ROVEListNew, 
                                                                                                                                                         ROVEsListNew, 
                                                                                                                                                         evaluator,
                                                                                                                                                         [sampleSize], 
                                                                                                                                                         kList, 
                                                                                                                                                         BList, 
                                                                                                                                                         k12List, 
                                                                                                                                                         B12List,
                                                                                                                                                         numReplicates)
            
            baseObjList.extend(baseObjListNew)
            MoVEObjList.extend(MoVEObjListNew)
            ROVEObjList.extend(ROVEObjListNew)
            ROVEsObjList.extend(ROVEsObjListNew)
            baseObjAvg.extend(baseObjAvgNew)
            MoVEObjAvg.extend(MoVEObjAvgNew)
            ROVEObjAvg.extend(ROVEObjAvgNew)
            ROVEsObjAvg.extend(ROVEsObjAvgNew)

            sampleSizeFinished.append(sampleSize)
            
            dumpEvalResults(baseObjList, 
                            MoVEObjList, 
                            ROVEObjList, 
                            ROVEsObjList, 
                            baseObjAvg, 
                            MoVEObjAvg, 
                            ROVEObjAvg, 
                            ROVEsObjAvg,
                            sampleSizeFinished, 
                            kList, 
                            BList, 
                            k12List, 
                            B12List,
                            numReplicates,
                            evalResultFile)
            
            plotAverage(baseObjAvg, 
                        MoVEObjAvg, 
                        ROVEObjAvg, 
                        ROVEsObjAvg,
                        sampleSizeFinished, 
                        kList, 
                        BList, 
                        k12List, 
                        B12List,
                        avgFigPath)
            
            plotCDF(baseObjList, 
                    MoVEObjList, 
                    ROVEObjList, 
                    ROVEsObjList, 
                    sampleSizeFinished, 
                    kList, 
                    BList, 
                    k12List, 
                    B12List,
                    cdfFigPath)
            
    except KilledByUser:
        logger.info("The experiment is killed")