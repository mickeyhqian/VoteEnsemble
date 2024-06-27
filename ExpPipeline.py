from Bagging import BaseTrainer, BAG, ReBAG
import numpy as np
from numpy.typing import NDArray
import time
import pickle
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Callable, Any



def runTraining(baseTrainer: BaseTrainer, 
                sampler: Callable[[int], NDArray],
                sampleSizeList: List[int], 
                kList: List[Tuple[int, float]], 
                BList: List[int], 
                k12List: List[Tuple[Tuple[int, float], Tuple[int, float]]], 
                B12List: List[Tuple[int, int]], 
                numReplicates: int):
    baseList = [[] for _ in range(len(sampleSizeList))]
    BAGList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(kList))] for _ in range(len(BList))]
    ReBAGList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(k12List))] for _ in range(len(B12List))]
    ReBAGSList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(k12List))] for _ in range(len(B12List))]

    for i in range(len(sampleSizeList)):
        n = sampleSizeList[i]
        for j in range(numReplicates):
            sample = sampler(n)
            
            tic = time.time()
            baseResult = baseTrainer.train(sample)
            if baseResult is None:
                raise RuntimeError(f"base training failed for sample size {n}, replication {j}")
            baseList[i].append(baseTrainer.toPickleable(baseResult))
            print(f"Finish base training for sample size {n}, replication {j}, taking {time.time()-tic} secs")

            for ind1, B in enumerate(BList):
                for ind2, (base, ratio) in enumerate(kList):
                    k = max(base, int(n * ratio))
                    bag = BAG(baseTrainer, numParallelTrain = 1, randomState = 666)

                    tic = time.time()
                    BAGList[ind1][ind2][i].append(baseTrainer.toPickleable(bag.run(sample, k, B)))
                    print(f"Finish BAG training for sample size {n}, replication {j}, B={B}, k={k}, taking {time.time()-tic} secs")

            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n * ratio1))
                    k2 = max(base2, int(n * ratio2))
                    rebag = ReBAG(baseTrainer, False, numParallelEval = 12, numParallelTrain = 1, randomState = 666)
                    
                    tic = time.time()
                    ReBAGList[ind1][ind2][i].append(baseTrainer.toPickleable(rebag.run(sample, k1, k2, B1, B2)))
                    print(f"Finish ReBAG training for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2 = {k2}, taking {time.time()-tic} secs")

            for ind1, (B1, B2) in enumerate(B12List):
                for ind2, ((base1, ratio1), (base2, ratio2)) in enumerate(k12List):
                    k1 = max(base1, int(n / 2 * ratio1))
                    k2 = max(base2, int(n / 2 * ratio2))
                    rebags = ReBAG(baseTrainer, True, numParallelEval = 12, numParallelTrain = 1, randomState = 666)

                    tic = time.time()
                    ReBAGSList[ind1][ind2][i].append(baseTrainer.toPickleable(rebags.run(sample, k1, k2, B1, B2)))
                    print(f"Finish ReBAGS training for sample size {n}, replication {j}, B1={B1}, B2={B2}, k1={k1}, k2 = {k2}, taking {time.time()-tic} secs")

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
    
    baseObjList = [[] for _ in range(len(sampleSizeList))]
    baseObjAvg = []
    BAGObjList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(kList))] for _ in range(len(BList))]
    BAGObjAvg = [[[] for _ in range(len(kList))] for _ in range(len(BList))]
    ReBAGObjList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(k12List))] for _ in range(len(B12List))]
    ReBAGObjAvg = [[[] for _ in range(len(k12List))] for _ in range(len(B12List))]
    ReBAGSObjList = [[[[] for _ in range(len(sampleSizeList))] for _ in range(len(k12List))] for _ in range(len(B12List))]
    ReBAGSObjAvg = [[[] for _ in range(len(k12List))] for _ in range(len(B12List))]

    for i in range(len(sampleSizeList)):
        for j in range(numReplicates):
            baseObjList[i].append(evaluator(baseTrainer.fromPickleable(baseList[i][j])))

            for ind1 in range(len(BList)):
                for ind2 in range(len(kList)):
                    BAGObjList[ind1][ind2][i].append(evaluator(baseTrainer.fromPickleable(BAGList[ind1][ind2][i][j])))

            for ind1 in range(len(B12List)):
                for ind2 in range(len(k12List)):
                    ReBAGObjList[ind1][ind2][i].append(evaluator(baseTrainer.fromPickleable(ReBAGList[ind1][ind2][i][j])))
                    ReBAGSObjList[ind1][ind2][i].append(evaluator(baseTrainer.fromPickleable(ReBAGSList[ind1][ind2][i][j])))
    
        if len(baseObjList[i]) > 0:
            baseObjAvg.append(np.mean(baseObjList[i]))
        for ind1 in range(len(BList)):
            for ind2 in range(len(kList)):
                if len(BAGObjList[ind1][ind2][i]) > 0:
                    BAGObjAvg[ind1][ind2].append(np.mean(BAGObjList[ind1][ind2][i]))
        for ind1 in range(len(B12List)):
            for ind2 in range(len(k12List)):
                if len(ReBAGObjList[ind1][ind2][i]) > 0:
                    ReBAGObjAvg[ind1][ind2].append(np.mean(ReBAGObjList[ind1][ind2][i]))
                if len(ReBAGSObjList[ind1][ind2][i]) > 0:
                    ReBAGSObjAvg[ind1][ind2].append(np.mean(ReBAGSObjList[ind1][ind2][i]))

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
                filePath: str):
    fig, ax = plt.subplots()
    ax.plot(sampleSizeList, baseObjAvg, marker = 'o', markeredgecolor = 'none', color = 'blue', linestyle = 'solid', linewidth = 2, label = 'base')

    for ind1, B in enumerate(BList):
        for ind2, k in enumerate(kList):
            ax.plot(sampleSizeList, BAGObjAvg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'BAG, B={B}, k={k}')
    
    for ind1, B12 in enumerate(B12List):
        for ind2, k in enumerate(k12List):
            ax.plot(sampleSizeList, ReBAGObjAvg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'ReBAG, B12={B12}, k={k}')
            ax.plot(sampleSizeList, ReBAGSObjAvg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'ReBAGS, B12={B12}, k={k}')
    
    ax.set_xlabel('sample size', size = 20)
    ax.set_ylabel('cost', size = 20)
    ax.legend(fontsize = 'small')
    fig.savefig(filePath, dpi=600)


def plotCDF(baseObjList: List, 
            BAGObjList: List, 
            ReBAGObjList: List, 
            ReBAGSObjList: List, 
            sampleSizeList: List, 
            kList: List, 
            BList: List, 
            k12List: List, 
            B12List: List,
            filePath: str):
    fig, ax = plt.subplots(nrows=len(sampleSizeList), figsize=(6, len(sampleSizeList) * 4))

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
        ax[i].plot(xList, yList, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'base')
        for ind1, B in enumerate(BList):
            for ind2, k in enumerate(kList):
                xList, yList = getCDF(BAGObjList[ind1][ind2][i])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'BAG, B={B}, k={k}')
        
        for ind1, B12 in enumerate(B12List):
            for ind2, k in enumerate(k12List):
                xList, yList = getCDF(ReBAGObjList[ind1][ind2][i])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'ReBAG, B12={B12}, k={k}')
                xList, yList = getCDF(ReBAGSObjList[ind1][ind2][i])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'ReBAGS, B12={B12}, k={k}')
        
        if i == len(sampleSizeList) - 1:
            ax[i].set_xlabel('cost')

        ax[i].set_ylabel('tail prob')
        ax[i].set_yscale('log')
        ax[i].set_title(f'sample size = {sampleSizeList[i]}', fontsize = "medium")

    # Create a legend using the first subplot
    handles, labels = ax[0].get_legend_handles_labels()

    # Place the combined legend outside the subplots
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize = 'small')
    fig.savefig(filePath, dpi=600)


def pipeline(caseName: str,
             expID: str,
             baseTrainer: BaseTrainer, 
             sampler: Callable,
             evaluator: Callable,
             sampleSizeList: List, 
             kList: List, 
             BList: List, 
             k12List: List, 
             B12List: List, 
             numReplicates: int):
    baseList, BAGList, ReBAGList, ReBAGSList = runTraining(baseTrainer, 
                                                           sampler,
                                                           sampleSizeList, 
                                                           kList, 
                                                           BList, 
                                                           k12List, 
                                                           B12List, 
                                                           numReplicates)
    scriptDir = os.path.dirname(__file__)
    trainingResultFile = f"{scriptDir}/ExpData/{caseName}/{expID}/trainingResults.pkl"
    dumpTrainingResults(baseList, 
                        BAGList, 
                        ReBAGList, 
                        ReBAGSList,
                        sampleSizeList, 
                        kList, 
                        BList, 
                        k12List, 
                        B12List, 
                        numReplicates,
                        trainingResultFile)
    

    baseObjList, BAGObjList, ReBAGObjList, ReBAGSObjList, baseObjAvg, BAGObjAvg, ReBAGObjAvg, ReBAGSObjAvg = runEvaluation(baseTrainer,
                                                                                                                           baseList, 
                                                                                                                           BAGList, 
                                                                                                                           ReBAGList, 
                                                                                                                           ReBAGSList, 
                                                                                                                           evaluator,
                                                                                                                           sampleSizeList, 
                                                                                                                           kList, 
                                                                                                                           BList, 
                                                                                                                           k12List, 
                                                                                                                           B12List,
                                                                                                                           numReplicates)
    
    evalResultFile = f"{scriptDir}/ExpData/{caseName}/{expID}/evalResults.pkl"
    dumpEvalResults(baseObjList, 
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
                    evalResultFile)
    

    avgFigPath = f"{scriptDir}/ExpData/{caseName}/{expID}/avgPlot.png"
    cdfFigPath = f"{scriptDir}/ExpData/{caseName}/{expID}/cdfPlot.png"
    plotAverage(baseObjAvg, 
                BAGObjAvg, 
                ReBAGObjAvg, 
                ReBAGSObjAvg,
                sampleSizeList, 
                kList, 
                BList, 
                k12List, 
                B12List,
                avgFigPath)
    
    plotCDF(baseObjList, 
            BAGObjList, 
            ReBAGObjList, 
            ReBAGSObjList, 
            sampleSizeList, 
            kList, 
            BList, 
            k12List, 
            B12List,
            cdfFigPath)