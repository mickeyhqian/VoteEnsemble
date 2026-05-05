import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from ExpPipeline import loadResults, plotAvgWithError, plotCDF

def plotSyntheticExp(expDir: str):
    expName = os.path.basename(expDir)
    resultPath = os.path.join(expDir, "evalResults.pkl")
    (
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
    ) = loadResults(resultPath)
    baggingObjList = []
    
    # if expDir.endswith("NN_d30_l4_ES_WB"):
    if False:
        fileList = [
            "1024_2048.pkl",
            "4096_8192.pkl",
            "16384_32768.pkl",
            "65536.pkl",
        ]
        data = {}
        for file in fileList:
            with open(os.path.join(expDir, file), "rb") as f:
                data.update(pickle.load(f))
        
        for size in sorted(data):
            valueDict = data[size]
            baggingObjList.append([valueDict[idx] for idx in sorted(valueDict)])

    plotAvgWithError(
        baseObjList,
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in MoVEObjList],
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in ROVEObjList],
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in ROVEsObjList],
        baggingObjList,
        numReplicates,
        0.95,
        sampleSizeList,
        os.path.join(expDir, f"{expName}_plots/{expName}_avg.png"),
        yLogScale=False
    )
    plotAvgWithError(
        baseObjList,
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in MoVEObjList],
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in ROVEObjList],
        [entry[0][0] if len(entry) > 0 and len(entry[0]) > 0 else [] for entry in ROVEsObjList],
        baggingObjList,
        numReplicates,
        0.95,
        sampleSizeList,
        os.path.join(expDir, f"{expName}_plots/{expName}_avg_ylog.png"),
        yLogScale=True
    )

    # if expDir.endswith("NN_d30_l4_ES_WB"):
    for i in range(len(sampleSizeList)):
        plotCDF(
            baseObjList[i],
            MoVEObjList[i][0][0] if len(MoVEObjList[i]) > 0 and len(MoVEObjList[i][0]) > 0 else [],
            ROVEObjList[i][0][0] if len(ROVEObjList[i]) > 0 and len(ROVEObjList[i][0]) > 0 else [],
            ROVEsObjList[i][0][0] if len(ROVEsObjList[i]) > 0 and len(ROVEsObjList[i][0]) > 0 else [],
            baggingObjList[i] if len(baggingObjList) > 0 else [],
            os.path.join(expDir, f"{expName}_plots/{expName}_cdf_{int(np.log2(sampleSizeList[i]))}.png"),
            xLogScale = False,
            yLogScale = True
        )
        plotCDF(
            baseObjList[i],
            MoVEObjList[i][0][0] if len(MoVEObjList[i]) > 0 and len(MoVEObjList[i][0]) > 0 else [],
            ROVEObjList[i][0][0] if len(ROVEObjList[i]) > 0 and len(ROVEObjList[i][0]) > 0 else [],
            ROVEsObjList[i][0][0] if len(ROVEsObjList[i]) > 0 and len(ROVEsObjList[i][0]) > 0 else [],
            baggingObjList[i] if len(baggingObjList) > 0 else [],
            os.path.join(expDir, f"{expName}_plots/{expName}_cdf_{int(np.log2(sampleSizeList[i]))}_xlog.png"),
            xLogScale = True,
            yLogScale = True
        )
        
if __name__ == "__main__":
    plotSyntheticExp("/home/hqian/ResearchProjects/VoteEnsemble/ExpData/NetworkData")