import numpy as np
from gurobipy import Model, GRB, quicksum
from multiprocessing import Queue, Process
import time
from utils.generateSamples import genSample_SSKP

############# Utility functions for parallel computation #############
def sequentialSolve(opt_func, queue: Queue, sampleList, *prob_args):
    for sample in sampleList:
        queue.put(opt_func(sample, *prob_args))

def sequentialEvaluate(eval_func, queue: Queue, taskList, *prob_args):
        # taskList: a list of tuples, tuple[0] is the sample and tuple[1] is the solution
        for task in taskList:
            queue.put(eval_func(task[0], task[1], *prob_args))

def sequentialRanking(eval_func, queue: Queue, taskList, *prob_args):
    # taskList: a list of tuples, tuple[0] is the sample, tuple[1] is the retrieved solutions, tuple[2] is the epsilon
    for task in taskList:
        queue.put(solution_ranking(task[0], task[1], task[2], eval_func, *prob_args))


############# Algorithm 1 #############
def majority_vote(sample_n, B, k, opt_func, rng, *prob_args):
    x_count = {}
    numProcesses = 8
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # print("solving")
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[rng.choice(sample_n.shape[0], k, replace=False)])
        processIndex = (processIndex + 1) % numProcesses
        # x_k = tuple(opt_func(sample_k, *prob_args))
        # x_count[x_k] = x_count.get(x_k, 0) + 1

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            processList.append(Process(target=sequentialSolve, args=(opt_func, queue, sampleLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()

    for _ in range(B):
        x_k = queue.get()
        if x_k is not None:
            sol = x_k if type(x_k) == int or type(x_k) == float else tuple(int(entry) for entry in x_k)
            x_count[sol] = x_count.get(sol, 0) + 1

    for process in processList:
        process.join()
    
    x_max = max(x_count, key=x_count.get)
    return x_max, list(x_count.keys())


############# Algorithms 3 and 4 #############
def solution_ranking(sample_k, retrieved_solutions, epsilon, eval_func, *prob_args):
    # this function supports solution ranking based on the maximization problem
    # input: epsilon should be numeric valued
    # output: epsilon-optimal solutions in the retrieved_solutions
    sol_evaluation = {}
    for sol in retrieved_solutions:
        obj = eval_func(sample_k, sol, *prob_args)
        obj = obj[0] if len(obj) >= 2 else obj
        sol_evaluation[sol] = obj
    obj_max = max(sol_evaluation.values())
    X_epsilon = []
    for key in sol_evaluation:
        if sol_evaluation[key] >= obj_max - epsilon:
            X_epsilon.append(key)
    return X_epsilon


def baggingPhaseII(sample_n, retrieved_solutions, B2, k, epsilon, eval_func, rng, *prob_args):
    # implements phase 2 with a fixed numeric epsilon value, use parallel processing
    x_count = {}
    numProcesses = 8
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B2):
        sample_k = sample_n[rng.choice(sample_n.shape[0], k, replace=False)]
        taskLists[processIndex].append((sample_k, retrieved_solutions, epsilon))
        processIndex = (processIndex + 1) % numProcesses

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            processList.append(Process(target=sequentialRanking, args = (eval_func, queue, taskLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()
    
    for _ in range(B2):
        X_epsilon = queue.get()
        if X_epsilon is not None:
            for x in X_epsilon:
                x_count[x] = x_count.get(x, 0) + 1
    
    for process in processList:
        process.join()

    return max(x_count, key=x_count.get), x_count

def baggingPhaseII_epsilonDynamic(sample_n, retrieved_solutions, B2, k, tolerance, eval_func, rng, *prob_args):
    # implement the strategy of dynamic epsilon, use the bisection method to find the proper epsilon value
    
    # boundary case: epsilon = 0
    x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, 0, eval_func, rng, *prob_args)
    if x_count[x_max] >= B2/2:
        return x_max, x_count, 0
    
    left, right = 0, 1
    x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, right, eval_func, rng, *prob_args)
    while x_count[x_max] < B2/2:
        left = right
        right *= 2
        x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, right, eval_func, rng, *prob_args)

    while right - left > tolerance:
        mid = (left + right) / 2
        x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, mid, eval_func, rng, *prob_args)
        if x_count[x_max] >= B2/2:
            right = mid
        else:
            left = mid
    
    return x_max, x_count, right

def baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args):
    # implementation of Algorithm 3
    # epsilon: can either be a numerical value or "dynamic", which stands for using the bisection method to find a proper value
    _, retrieved_solutions = majority_vote(sample_n, B1, k, opt_func, rng, *prob_args)

    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions
    elif type(epsilon) is not str:
        x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, epsilon, eval_func, rng, *prob_args)
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max, x_count = baggingPhaseII(sample_n, retrieved_solutions, B2, k, 0, eval_func, rng, *prob_args)
    else: # epsilon == "dynamic" and more than two solutions
        x_max, x_count, _ = baggingPhaseII_epsilonDynamic(sample_n, retrieved_solutions, B2, k, tolerance, eval_func, rng, *prob_args)
        
    return x_max, x_count, retrieved_solutions
    

def baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args):
    # implementation of Algorithm 4
    sample_n1 = sample_n[:int(sample_n.shape[0]/2)]
    sample_n2 = sample_n[int(sample_n.shape[0]/2):]

    _, retrieved_solutions = majority_vote(sample_n1, B1, k, opt_func, rng, *prob_args)
    
    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions
    elif type(epsilon) is not str:
        eps = epsilon
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        eps = 0
    else: # epsilon == "dynamic" and more than two solutions
        _, _, eps = baggingPhaseII_epsilonDynamic(sample_n1, retrieved_solutions, B2, k, tolerance, eval_func, rng, *prob_args)
        
    x_max, x_count = baggingPhaseII(sample_n2, retrieved_solutions, B2, k, eps, eval_func, rng, *prob_args)
    return x_max, x_count, retrieved_solutions


############# Maximum weight matching #############
def gurobi_matching(sample_k, N, w):
    # sample_k: k * 6 matrix, where each column corresponds the weight of an edge
    # N: number of nodes >= 4.
    # w: dictionary of edge weights, using 1-based index
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(1,4):
        for j in range(i+1, 5):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    model = Model("max_weight_matching")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(w[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(x[(min(i, j), max(i, j))] for j in range(1, N+1) if i != j) <= 1 for i in range(1, N+1))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[e].X) for e in edges])
        return x_opt #, x.keys()
    else:
        print("No optimal solution found.")
        return None    


############# Newsvendor #############

def solveNewsVendorSAA(sample_xi, price, cost):
    sortedData = np.sort(sample_xi)
    q = (price - cost) / price
    idx = int(q * len(sample_xi))
    return 0.1*int(np.ceil(10*sortedData[idx]))

############# SSKP #############

def prob_simulate_SSKP(n, num_repeats, rng_sample, sample_args, *prob_args):
    # simulate the probability \hat p(x) for a given n
    # the difference from the majority_vote function is that this function uses new samples each time
    count = {}
    numProcesses = 8
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(num_repeats):
        sample = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
        sampleLists[processIndex].append(sample)
        processIndex = (processIndex + 1) % numProcesses

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            processList.append(Process(target=sequentialSolve, args=(gurobi_SSKP, queue, sampleLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()
    
    for _ in range(num_repeats):
        x = queue.get()
        if x is not None:
            sol = x if type(x) == int else tuple(int(entry) for entry in x)
            count[sol] = count.get(sol, 0) + 1

    for process in processList:
        process.join()
    
    for key in count:
        count[key] /= num_repeats
    
    return count


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
    # model.setParam('MIPGap', 0.00001)
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")

    model.setObjective(quicksum(r[i] * x[i] for i in range(m)) - c/k * quicksum(z[j] for j in range(k)), GRB.MAXIMIZE)
    model.addConstrs((z[j] >= quicksum(sample_k[j,i] * x[i] for i in range(m)) - q for j in range(k)), name="budget")
    model.setParam("Threads", 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[i].X) for i in range(m)])
        # z_opt = np.array([z[j].X for j in range(k)]) # no need to output
        return x_opt
    else:
        print("No optimal solution found.")
        return None
    
############# Portfolio #############
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
    model.setParam('MIPGap', 0.00001)
    x = model.addVars(m, lb=0, vtype=GRB.INTEGER, name="x")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")
    
    model.setObjective(c + 1/0.05 * 1/k * quicksum(z[i] for i in range(k)), GRB.MINIMIZE)
    
    model.addConstrs(z[i] >= -sum(sample_xi[i, j] * x[j] * p[j] for j in range(m)) - c for i in range(k))
    model.addConstr(quicksum(mu[j] * x[j] * p[j] for j in range(m)) >= 1.5*b)
    model.addConstr(quicksum(x[j] * p[j] for j in range(m)) <= b)
    model.setParam("Threads", 3)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[j].X) for j in range(m)])
        return x_opt
    else:
        print("No optimal solution found.")
        return None


############# Network #############
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
        x_opt = np.array([round(x[i].X) for i in range(p)])
        # y_sp_opt = np.array([[[[y_sp[i, j, l, a].X for a in range(k)] for l in range(g)] for j in range(p)] for i in range(s)])
        # y_pc_opt = np.array([[[[y_pc[j, i, l, a].X for a in range(k)] for l in range(g)] for j in range(p)] for i in range(c)])
        # z_opt = np.array([[[z[i, l, a].X for a in range(k)] for l in range(g)] for i in range(c)])
        return x_opt #, y_sp_opt, y_pc_opt, z_opt
    else:
        print("No optimal solution found.")
        return None

############# Algorithm function for Network #############
def sequentialSolve_network(eval_func, queue: Queue, sampleList, *eval_args):
    for sample in sampleList:
        queue.put(eval_func(sample[0],sample[1], *eval_args))

def majority_vote_network(sample_S, sample_D, B, k, eval_func, rng, *eval_args):
    x_count = {}
    numProcesses = 8
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


def gurobi_second_stage_wSol(sample, x, C, Q_sp, Q_pc, R, M, H):
    # second stage LP problem
    tic = time.time()
    sample_S, sample_D = sample[0], sample[1]
    s, p, g = Q_sp.shape
    c, g, n = sample_D.shape
    model = Model("second_stage")
    model.setParam(GRB.Param.OutputFlag, 0)
    y_sp = model.addVars(s, p, g, n, lb=0, vtype=GRB.CONTINUOUS, name="y_sp")
    y_pc = model.addVars(p, c, g, n, lb=0, vtype=GRB.CONTINUOUS, name="y_pc")
    z = model.addVars(c, g, n, lb=0, vtype=GRB.CONTINUOUS, name="z")

    obj_expr = 1/n * quicksum(Q_sp[i, j, l] * y_sp[i, j, l, a] for i in range(s) for j in range(p) for l in range(g) for a in range(n))\
                        + 1/n * quicksum(Q_pc[j, i, l] * y_pc[j, i, l, a] for j in range(p) for i in range(c) for l in range(g) for a in range(n))\
                            + 1/n * quicksum(H[i, l] * z[i, l, a] for i in range(c) for l in range(g) for a in range(n))
    
    model.setObjective(obj_expr, GRB.MINIMIZE)

    model.addConstrs((quicksum(y_sp[i, j, l, a] for i in range(s)) - quicksum(y_pc[j, i, l, a] for i in range(c)) == 0
                        for a in range(n) for l in range(g) for j in range(p)), name="flow")
    
    model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sample_D[i, l, a]
                        for a in range(n) for l in range(g) for i in range(c)), name="demand")
    
    model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sample_S[i, l, a]
                        for a in range(n) for l in range(g) for i in range(s)), name="supply")
    
    model.addConstrs((quicksum(R[j, l] * quicksum(y_sp[i, j, l, a] for i in range(s)) for l in range(g)) <= M[j] * x[j]
                        for a in range(n) for j in range(p)), name="capacity")
    
    model.setParam("Threads", 9)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Time elapsed: {time.time()-tic}")
        return model.ObjVal + sum(C[j] * x[j] for j in range(p)), x
    else:
        print("No optimal solution found.")
        return None


