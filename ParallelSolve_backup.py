import numpy as np
from gurobipy import Model, GRB, quicksum
from multiprocessing import Queue, Process


# gurobi function that solves the sample-based SSKP problem with given parameters
def gurobi_SSKP(sample_k, r, c, q):
    # input:
    # sample_k (weight matrix): 2d numpy array of shape (k, m), where m is the number of items, k is the number of samples
    # r (reward vector): 1d numpy array of shape (m, )
    # c (cost vector): scalar
    # q (budget): scalar
    # output: 
    # x (selection vector): 1d numpy array of shape (m, )
    k, m = sample_k.shape
    model = Model("SSKP")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")

    model.setObjective(quicksum(r[i] * x[i] for i in range(m)) - c/k * quicksum(z[j] for j in range(k)), GRB.MAXIMIZE)
    model.addConstrs((z[j] >= quicksum(sample_k[j,i] * x[i] for i in range(m)) - q for j in range(k)), name="budget")
    model.setParam("Threads", 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[i].X for i in range(m)])
        z_opt = np.array([z[j].X for j in range(k)]) # no need to output
        return x_opt
    else:
        print("No optimal solution found.")
        return None
    

def gurobi_portfolio(sample_xi, p, mu, b):
    # input:
    # sample_xi(random returns): k*m numpy array (where m is number of assets)
    # p(price): m*1 numpy array
    # mu(mean return): m*1 numpy array
    # b(budget): 1*1
    # output:
    # x(number of shares for each asset): m*1 numpy array

    k, m = sample_xi.shape
    model = Model("portfolio")
    model.setParam(GRB.Param.OutputFlag, 0)
    x = model.addVars(m, lb=0, vtype=GRB.INTEGER, name="x")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")
    
    model.setObjective(c + 1/0.05 * 1/k * quicksum(z[i] for i in range(k)), GRB.MINIMIZE)
    
    model.addConstrs(z[i] >= -sum(sample_xi[i, j] * x[j] * p[j] for j in range(m)) - c for i in range(k))
    model.addConstr(quicksum(mu[j] * x[j] * p[j] for j in range(m)) >= 1.5*b)
    model.addConstr(quicksum(x[j] * p[j] for j in range(m)) <= b)
    model.setParam("Threads", 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(m)])
        return x_opt
    else:
        print("No optimal solution found.")
        return None



def sequentialSolve(eval_func, queue: Queue, sampleList, *eval_args):
    for sample in sampleList:
        x_k = tuple(eval_func(sample, *eval_args))
        queue.put(x_k)


def majority_vote(sample_n, B, k, eval_func, *eval_args):
    x_count = {}
    numProcesses = 5
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[np.random.choice(sample_n.shape[0], k, replace=False)])
        processIndex += 1
        if processIndex >= numProcesses:
            processIndex = 0
        # x_k = tuple(eval_func(sample_k, *eval_args))
        # x_count[x_k] = x_count.get(x_k, 0) + 1
            
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            processList.append(Process(target=sequentialSolve, args=(eval_func, queue, sampleLists[i], *eval_args), daemon = True))
    
    for process in processList:
        process.start()

    for process in processList:
        process.join()

    while not queue.empty():
        x_k = queue.get()
        x_count[x_k] = x_count.get(x_k, 0) + 1
    
    x_max = max(x_count, key=x_count.get)
    return x_max