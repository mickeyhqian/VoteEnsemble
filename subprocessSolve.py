import pickle
import sys
from GurobiSolve import gurobi_portfolio, gurobi_SSKP


if __name__ == "__main__":
    problemName = sys.argv[1]
    filePath = sys.argv[2]

    with open(filePath, "rb") as f:
        dataIn = pickle.load(f)

    solList = []
    if problemName == "portfolio":
        for data in dataIn[0]:
            solList.append(gurobi_portfolio(data, *dataIn[1:]))
    elif problemName == "sskp":
        for data in dataIn[0]:
            solList.append(gurobi_SSKP(data, *dataIn[1:]))
    
    with open(filePath, "wb") as f:
        pickle.dump(solList, f)