from Examples import BaseLR
from ExpUtils import runTraining, runEvaluation, dumpEvalResults, dumpTrainingResults, loadResults, plotAverage, plotCDF
import numpy as np
from multiprocessing import set_start_method
import os
from uuid import uuid4


if __name__ == "__main__":
    if os.name == "posix":
        set_start_method("fork")
        
    caseName = "LR_SAA"

    rngData = np.random.default_rng(seed = 888)
    rngProb = np.random.default_rng(seed = 999)

    meanX = rngProb.uniform(1.1, 1.9, 10)
    beta = rngProb.uniform(1, 20, 10)
    noiseShape = 2.1

    lr = BaseLR(meanX, beta, noiseShape)

    kList = []
    BList = []
    k12List = [((30, 0.5), (30, 0.005))]
    B12List = [(50, 200)]
    numReplicates = 10
    sampleSizeList = [2**i for i in range(10, 15)]

    def sampler(n: int):
        return lr.genSample(n, rngData)
    
    def evaluator(trainingResult):
        return lr.optimalityGap(trainingResult)
    
    baseList, BAGList, ReBAGList, ReBAGSList = runTraining(lr, 
                                                           sampler,
                                                           sampleSizeList, 
                                                           kList, 
                                                           BList, 
                                                           k12List, 
                                                           B12List, 
                                                           numReplicates)
    
    expID = str(uuid4())
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
    

    baseObjList, BAGObjList, ReBAGObjList, ReBAGSObjList, baseObjAvg, BAGObjAvg, ReBAGObjAvg, ReBAGSObjAvg = runEvaluation(lr,
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