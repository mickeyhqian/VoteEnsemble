from BaseLearners import BaseNN, RegressionNN
import numpy as np
from numpy.typing import NDArray
from multiprocessing import set_start_method
from uuid import uuid4
import pickle
import json
import sys
import os
import torch
import logging
logger = logging.getLogger(name = "VE")



if __name__ == "__main__":
    set_start_method("spawn")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    if len(sys.argv) > 1:
        resultDir = sys.argv[1]
    else:
        resultDir = os.path.join(os.path.dirname(__file__), str(uuid4()))

    sampleSizeList = [65536]
    caseName = "_".join([str(entry) for entry in sampleSizeList])

    os.makedirs(resultDir, exist_ok = True)
    logger.setLevel(logging.DEBUG)
    logHandler = logging.FileHandler(os.path.join(resultDir, f"bagging_{caseName}.log"))
    formatter = logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(message)s")
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    rngEval = np.random.default_rng(seed = 777)

    d = 30
    meanX = np.linspace(1, 100, num = d)

    def trueMapping(x: NDArray) -> float:
        return np.mean(np.log(x + 1))
    
    def evalSampler(n: int) -> NDArray:
        XSample = rngEval.uniform(low = 0, high = 2 * meanX, size = (n, len(meanX)))
        YSample = np.asarray([[trueMapping(x)] for x in XSample])
        return np.hstack((YSample, XSample))
    
    # baseNN = BaseNN([50, 300, 500, 800, 800, 500, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 300, 500, 500, 300, 50], learningRate = 0.001, useGPU = False)
    baseNN = BaseNN([50, 300, 300, 50], learningRate = 0.001, useGPU = False)
    # baseNN = BaseNN([50, 50], learningRate = 0.001, useGPU = False)

    evalSample = evalSampler(1000000)
    evalDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def inference(learningResult: RegressionNN) -> torch.Tensor:
        return baseNN.inference(learningResult, evalSample, device = evalDevice)

    modelDir = os.path.join(resultDir, "subsampleResults")
    if not os.path.isdir(modelDir):
        raise ValueError("model directory not present")
    

    dumpFilePKL = os.path.join(resultDir, caseName + ".pkl")
    dumpFileJSON = os.path.join(resultDir, caseName + ".json")

    def getSizeAndIndex(dirName: str):
        infoList = dirName.split("_")
        return int(infoList[1]), int(infoList[2])
    
    def getSubsampleIndex(fileName: str):
        return int(fileName.split("_")[1])
    
    lossData = {}
    dirList = os.listdir(modelDir)
    criterion = torch.nn.MSELoss()
    YTruth = torch.Tensor(evalSample[:, :1])
    for dirName in dirList:
        dirPath = os.path.join(modelDir, dirName)
        sampleSize, index = getSizeAndIndex(dirName)
        if sampleSize not in sampleSizeList:
            continue

        if sampleSize not in lossData:
            lossData[sampleSize] = {}

        modelFiles = os.listdir(dirPath)
        bagPrediction = 0
        for modelFile in modelFiles:
            subsampleIndex = getSubsampleIndex(modelFile)
            modelPath = os.path.join(dirPath, modelFile)

            model = baseNN.loadLearningResult(modelPath)
            prediction = inference(model)
            bagPrediction += prediction

            # logger.info(f"prediction shape = {prediction.shape}")
            # logger.info(f"bagPrediction shape = {bagPrediction.shape}")
            
            logger.info(f"Finish evaluating conventional bagging for sample size = {sampleSize}, index = {index}, subsample index = {subsampleIndex}")
        
        bagPrediction /= len(modelFiles)
        lossData[sampleSize][index] = criterion(bagPrediction, YTruth).item()
        with open(dumpFilePKL, "wb") as f:
            pickle.dump(lossData, f)
        with open(dumpFileJSON, "w") as f:
            json.dump(lossData, f, indent = 4)
        logger.info(f"Finish evaluating conventional bagging for sample size = {sampleSize}, index = {index}, loss = {lossData[sampleSize][index]}")