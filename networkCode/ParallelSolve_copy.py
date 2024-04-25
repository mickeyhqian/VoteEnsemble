import numpy as np
from gurobipy import Model, GRB, quicksum
from multiprocessing import Queue, Process


def gurobi_network_first_stage(sample_S, sample_D, C, Q_sp, Q_pc, R, M, H):
    # input:
    # C (unit cost for building a facility): p*1 numpy array
    # Q_sp (unit flow cost): s*p*g numpy array
    # Q_pc (unit flow cost): p*c*g numpy array
    # sample_S (supply): s*g*k numpy array
    # sample_D (demand): c*g*k numpy array
    # R (unit processing require): p*g numpy array
    # M (processing capacity): p*1 numpy array
    # H (multiplier): c*g numpy array
    # output:
    # x (open facility): p*1 numpy array
    # y_sp (flow supplier --> facility): s*p*g*k numpy array 
    # y_pc (flow facility --> consumer): p*c*g*k numpy array
    # z (multiplier): c*g*k numpy array
    s, p, g = Q_sp.shape
    c, g, k = sample_D.shape
    model = Model("network")
    model.setParam(GRB.Param.OutputFlag, 0)
    x = model.addVars(p, vtype=GRB.BINARY, name="x")
    y_sp = model.addVars(s, p, g, k, lb=0, vtype=GRB.CONTINUOUS, name="y_sp")
    y_pc = model.addVars(p, c, g, k, lb=0, vtype=GRB.CONTINUOUS, name="y_pc")
    z = model.addVars(c, g, k, lb=0, vtype=GRB.CONTINUOUS, name="z")

    obj_expr = quicksum(C[j] * x[j] for j in range(p)) \
                       + 1/k * quicksum(Q_sp[i, j, l] * y_sp[i, j, l, a] for i in range(s) for j in range(p) for l in range(g) for a in range(k))\
                        + 1/k * quicksum(Q_pc[j, i, l] * y_pc[j, i, l, a] for j in range(p) for i in range(c) for l in range(g) for a in range(k))\
                            + 1/k * quicksum(H[i, l] * z[i, l, a] for i in range(c) for l in range(g) for a in range(k))
    
    model.setObjective(obj_expr, GRB.MINIMIZE)

    model.addConstrs((quicksum(y_sp[i, j, l, a] for i in range(s)) - quicksum(y_pc[j, i, l, a] for i in range(c)) == 0 
                      for a in range(k) for l in range(g) for j in range(p)), name="flow")
    
    model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sample_D[i, l, a]
                      for a in range(k) for l in range(g) for i in range(c)), name="demand")
    
    model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sample_S[i, l, a]
                      for a in range(k) for l in range(g) for i in range(s)), name="supply")
    
    model.addConstrs((quicksum(R[j, l] * quicksum(y_sp[i, j, l, a] for i in range(s)) for l in range(g)) <= M[j] * x[j]
                      for a in range(k) for j in range(p)), name="capacity")

    model.setParam("Threads", 1)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # only return the discrete variable x
        x_opt = np.array([x[i].X for i in range(p)])
        # y_sp_opt = np.array([[[[y_sp[i, j, l, a].X for a in range(k)] for l in range(g)] for j in range(p)] for i in range(s)])
        # y_pc_opt = np.array([[[[y_pc[j, i, l, a].X for a in range(k)] for l in range(g)] for j in range(p)] for i in range(c)])
        # z_opt = np.array([[[z[i, l, a].X for a in range(k)] for l in range(g)] for i in range(c)])
        return x_opt #, y_sp_opt, y_pc_opt, z_opt
    else:
        print("No optimal solution found.")
        return None
    

def sequentialSolve_network(eval_func, queue: Queue, sampleList, *eval_args):
    for sample in sampleList:
        queue.put(eval_func(sample[0],sample[1], *eval_args))

def majority_vote_network(sample_S, sample_D, B, k, eval_func, rng, *eval_args):
    x_count = {}
    numProcesses = 3
    sampleLists = [[] for _ in range(numProcesses)] # contain both sample_S and sample_D
    processIndex = 0
    for _ in range(B):
        sampleLists[processIndex].append(
            [sample_S[:,:,rng.choice(sample_S.shape[2], k, replace=False)], 
             sample_D[:,:,rng.choice(sample_D.shape[2], k, replace=False)]]
             )
        processIndex += 1
        if processIndex >= numProcesses:
            processIndex = 0

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            processList.append(Process(target=sequentialSolve_network, args=(eval_func, queue, sampleLists[i], *eval_args), daemon = True))
    
    for process in processList:
        process.start()

    for _ in range(B):
        x_k = queue.get()
        if x_k is not None:
            sol = tuple(int(entry) for entry in x_k)
            x_count[sol] = x_count.get(sol, 0) + 1

    for process in processList:
        process.join()
    
    x_max = max(x_count, key=x_count.get)
    return x_max