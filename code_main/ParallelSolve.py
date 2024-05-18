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
    # this evaluation function is used for final solution evaluation
    # taskList: a list of tuples, tuple[0] is the sample and tuple[1] is the solution
    for task in taskList:
        queue.put(eval_func(task[0], task[1], *prob_args))

def sequentialEvaluate_twoPhase(eval_func, queue: Queue, taskList, *prob_args):
    # this evaluation function is used for the second stage of bagging
    for task in taskList:
        queue.put(solution_evaluation(task[0], task[1], eval_func, *prob_args))

############# Algorithm 1 #############
def majority_vote(sample_n, B, k, opt_func, rng, *prob_args,varepsilon=None):
    x_count = {}
    numProcesses = 9
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[rng.choice(sample_n.shape[0], k, replace=False)])
        processIndex = (processIndex + 1) % numProcesses

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            args = (opt_func, queue, sampleLists[i]) + prob_args
            if varepsilon is not None:
                args += (varepsilon,)
            processList.append(Process(target=sequentialSolve, args=args, daemon=True))
            # processList.append(Process(target=sequentialSolve, args=(opt_func, queue, sampleLists[i], *prob_args), daemon = True))
    
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
def solution_evaluation(sample_k, retrieved_solutions, eval_func, *prob_args):
    # output: a numpy array of evaluations of the retrieved solutions
    # assume the problem is a maximization problem
    evaluations = []
    max_obj = -np.inf
    for sol in retrieved_solutions:
        obj = eval_func(sample_k, sol, *prob_args)
        obj = obj[0] if len(obj) >= 2 else obj
        max_obj = max(max_obj, obj)
        evaluations.append(obj)
    return np.array(evaluations), max_obj

def get_adaptive_epsilon(suboptimality_gap_matrix, tolerance):
    # implement the strategy of dynamic epsilon, use the bisection method to find the proper epsilon value
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, 0)
    if np.max(x_ratio) >= 0.5:
        return 0
    
    left, right = 0, 1
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, right)
    while np.max(x_ratio) < 0.5:
        left = right
        right *= 2
        x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, right)
    
    while right - left > tolerance:
        mid = (left + right) / 2
        x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, mid)
        if np.max(x_ratio) >= 0.5:
            right = mid
        else:
            left = mid
    
    return right

def get_suboptimality_gap(evaluation_matrix, max_obj_vector):
    # calculate the suboptimal gap, assume it is a maximization problem
    # simply use the max_obj_vector to minus the evaluation_matrix
    max_obj_vector = max_obj_vector.reshape(-1,1)
    return max_obj_vector - evaluation_matrix

def get_epsilon_optimal(suboptimality_gap_matrix, epsilon):
    # for the given suboptimal gap matrix, return the ratio of within epsilon-optimal
    # note that suboptimality_gap is calculated as max_obj - obj
    return np.sum(suboptimality_gap_matrix <= epsilon, axis=0)/suboptimality_gap_matrix.shape[0]

def get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon):
    # get the phase II solution based on the suboptimal gap matrix
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, epsilon)
    ind_max = np.argmax(x_ratio)
    return retrieved_solutions[ind_max]

def get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k, eval_func, rng, *prob_args):
    # generate a evaluation matrix, where i,j-th entry is the evaluation of the j-th solution in the i-th resample
    # i.e., each resample get a new row of evaluations
    # then, for each solution, calculate its suboptimal gap in each resample
    tic = time.time()
    evaluation_matrix = np.zeros((B2, len(retrieved_solutions)))
    max_obj_vector = np.zeros(B2)

    numProcesses = 9
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B2):
        sample_k = sample_n[rng.choice(sample_n.shape[0], k, replace=False)]
        taskLists[processIndex].append((sample_k, retrieved_solutions))
        processIndex = (processIndex + 1) % numProcesses
    
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            processList.append(Process(target=sequentialEvaluate_twoPhase, args = (eval_func, queue, taskLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()
    
    for iter in range(B2):
        evaluations_array, max_obj = queue.get()
        evaluation_matrix[iter] = evaluations_array
        max_obj_vector[iter] = max_obj

    for process in processList:
        process.join()

    suboptimality_gap_matrix = get_suboptimality_gap(evaluation_matrix, max_obj_vector)
    print(f"Time for generating suboptimality gap matrix: {time.time()-tic}")
    return suboptimality_gap_matrix

# Note: only Phase I is related to specific algorithms, e.g., SAA, DRO. Phase II is a general solution ranking stage.
# Note: retrieve_solutions is a list of solutions, so it is ordered
def baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, varepsilon = None):
    # implementation of Algorithm 3
    # epsilon: can either be a numerical value or "dynamic", which stands for using the bisection method to find a proper value
    _, retrieved_solutions = majority_vote(sample_n, B1, k, opt_func, rng, *prob_args, varepsilon=varepsilon)

    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    suboptimality_gap_matrix = get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, 0)
        epsilon = 0
    else: # epsilon == "dynamic" and more than two solutions
        epsilon = get_adaptive_epsilon(suboptimality_gap_matrix, tolerance)
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
        
    return x_max, suboptimality_gap_matrix, retrieved_solutions, epsilon
    

def baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, varepsilon = None):
    # implementation of Algorithm 4
    sample_n1 = sample_n[:int(sample_n.shape[0]/2)]
    sample_n2 = sample_n[int(sample_n.shape[0]/2):]

    _, retrieved_solutions = majority_vote(sample_n1, B1, k, opt_func, rng, *prob_args, varepsilon=varepsilon)
    
    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    suboptimality_gap_matrix_n2 = get_suboptimality_gap_matrix(sample_n2, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    if len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, 0)
        epsilon = 0
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    # epsilon == "dynamic" and more than two solutions
    suboptimality_gap_matrix_n1 = get_suboptimality_gap_matrix(sample_n1, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    epsilon = get_adaptive_epsilon(suboptimality_gap_matrix_n1, tolerance)
    x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
    return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon


############# LP version Algorithms #############
def check_exist(x_count, x):
    if x_count == {}:
        return None
    eps = 1e-8
    for key in x_count:
        if np.linalg.norm(np.array(key) - np.array(x)) < eps:
            return key
    return None
        

def majority_vote_LP(sample_n, B, k, opt_func, rng, *prob_args):
    x_count = {}
    numProcesses = 9
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[rng.choice(sample_n.shape[0], k, replace=False)])
        processIndex = (processIndex + 1) % numProcesses

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
            sol = x_k if type(x_k) == int or type(x_k) == float else tuple(entry for entry in x_k)
            key = check_exist(x_count, sol)
            if key is not None:
                x_count[key] += 1
            else:
                x_count[sol] = 1
    
    for process in processList:
        process.join()

    x_max = max(x_count, key=x_count.get)
    return x_max, list(x_count.keys())


# Note, functions baggingPhaseII and baggingPhaseII_epsilonDynamic need not to be modified for the LP version
def baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, k2 = None):
    _, retrieved_solutions = majority_vote_LP(sample_n, B1, k, opt_func, rng, *prob_args)
    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    if k2 is None:
        k2 = k

    suboptimality_gap_matrix = get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, 0)
        epsilon = 0
    else: # epsilon == "dynamic" and more than two solutions
        epsilon = get_adaptive_epsilon(suboptimality_gap_matrix, tolerance)
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    
    return x_max, suboptimality_gap_matrix, retrieved_solutions, epsilon


def baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, k2 = None):
    # implementation of Algorithm 4
    sample_n1 = sample_n[:int(sample_n.shape[0]/2)]
    sample_n2 = sample_n[int(sample_n.shape[0]/2):]

    _, retrieved_solutions = majority_vote_LP(sample_n1, B1, k, opt_func, rng, *prob_args)

    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    if k2 is None:
        k2 = k

    suboptimality_gap_matrix_n2 = get_suboptimality_gap_matrix(sample_n2, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    if len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, 0)
        epsilon = 0
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    # epsilon == "dynamic" and more than two solutions
    suboptimality_gap_matrix_n1 = get_suboptimality_gap_matrix(sample_n1, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    epsilon = get_adaptive_epsilon(suboptimality_gap_matrix_n1, tolerance)
    x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
    return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon


############# Algorithms for testing #############
def test_baggingTwoPhase_woSplit_LP(sample_n, B1, k, opt_func, eval_func, rng, *prob_args):
    # assume the problem is a minimization problem
    # here the eval_func is the exact evaluation function
    # retrieved_solutions is a list of tuples
    _, retrieved_solutions = majority_vote_LP(sample_n, B1, k, opt_func, rng, *prob_args)
    # compute the average of all solution vectors
    avg_sol = np.mean(retrieved_solutions, axis=0)
    avg_sol = tuple(float(entry) for entry in avg_sol)
    obj_avg_sol = eval_func(avg_sol, *prob_args)
        
    obj_dict = {sol: eval_func(sol, *prob_args) for sol in retrieved_solutions}
    # return the solution with the minimum objective value
    x_min = min(obj_dict, key=obj_dict.get)
    obj_min = obj_dict[x_min]
    
    x_min_avg = avg_sol if obj_avg_sol < obj_min else x_min
    obj_min_avg = obj_avg_sol if obj_avg_sol < obj_min else obj_min

    return x_min, obj_min, x_min_avg, obj_min_avg



############# Maximum weight matching #############
def gurobi_matching(sample_k, N, w):
    # maximum weight bipartite matching
    # sample_k: k * 9 matrix, where each column corresponds the weight of an edge
    # N: number of one-side nodes >= 6
    # w: dictionary of edge weights, using 0-based index
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    model = Model("max_weight_matching")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam("Method", 0) 
    edges = [(i, j) for i in range(N) for j in range(N)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(w[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(x[(i,j)] for j in range(N)) <= 1 for i in range(N))
    model.addConstrs(quicksum(x[(i,j)] for i in range(N)) <= 1 for j in range(N))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[e].X) for e in edges])
        # print([x[e].X for e in edges],x_opt, model.ObjVal)
        return x_opt #, x.keys()
    else:
        print("No optimal solution found.")
        return None    

def gurobi_matching_DRO_wasserstein(sample_k, N, w, varepsilon = None):
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind] - varepsilon
            ind += 1
    
    model = Model("max_weight_matching_DRO")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam("Method", 0) 
    edges = [(i, j) for i in range(N) for j in range(N)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(w[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(x[(i,j)] for j in range(N)) <= 1 for i in range(N))
    model.addConstrs(quicksum(x[(i,j)] for i in range(N)) <= 1 for j in range(N))

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
    numProcesses = 9
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

# TODO: need to place varepsilon to the last position
def gurobi_SSKP_DRO_wasserstein(sample_k, r, c, q, varepsilon = None):
    # this varepsilon is the threshold for the Wasserstein distance-based ambiguity set
    k, m = sample_k.shape
    model = Model("SSKP_DRO")
    model.setParam(GRB.Param.OutputFlag, 0)
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")

    # note that the only difference is a regularization term in the objective function
    model.setObjective(quicksum( (r[i] - c*varepsilon) * x[i] for i in range(m)) - c/k * quicksum(z[j] for j in range(k)), GRB.MAXIMIZE)
    model.addConstrs((z[j] >= quicksum(sample_k[j,i] * x[i] for i in range(m)) - q for j in range(k)), name="budget")
    model.setParam("Threads", 9)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[i].X) for i in range(m)])
        return x_opt
    else:
        print("No optimal solution found.")
        return None
 
############# Portfolio #############
def gurobi_portfolio(sample_xi, p, mu, b, alpha):
    # input:
    # sample_xi(random returns): k*m numpy array (where m is number of assets)
    # p(price): m*1 numpy array
    # mu(mean return): m*1 numpy array
    # b(budget): 1*1
    # output:
    # x(number of shares for each asset): m*1 numpy array

    # for consistency, use the maximization form

    k, m = sample_xi.shape
    model = Model("portfolio")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam('MIPGap', 0.00001)
    x = model.addVars(m, lb=0, vtype=GRB.INTEGER, name="x")
    c = model.addVar(lb=-float("inf"), vtype=GRB.CONTINUOUS, name="c")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")
    
    # model.setObjective(c + 1/(1-alpha) * 1/k * quicksum(z[i] for i in range(k)), GRB.MINIMIZE)
    model.setObjective(-c - 1/(1-alpha) * 1/k * quicksum(z[i] for i in range(k)), GRB.MAXIMIZE)
    
    model.addConstrs(z[i] >= -sum(sample_xi[i, j] * x[j] * p[j] for j in range(m)) - c for i in range(k))
    model.addConstr(quicksum(mu[j] * x[j] * p[j] for j in range(m)) >= 1.5*b)
    model.addConstr(quicksum(x[j] * p[j] for j in range(m)) <= b)
    model.setParam("Threads", 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([round(x[j].X) for j in range(m)])
        # objVal = model.ObjVal
        return x_opt
    else:
        print("No optimal solution found.")
        return None

############# Continuous portfolio #############
def gurobi_portfolio_continuous(sample_xi, mu, b):
    k, m = sample_xi.shape # k is the number of samples
    mu = np.array(mu).reshape(1,-1)
    sample_xi_adjusted = sample_xi - mu

    model = Model("portfolio_continuous")
    model.setParam(GRB.Param.OutputFlag, 0)

    x = model.addMVar(shape=m, lb=0, name = "x")
    deviations = sample_xi_adjusted @ x
    avg_squared_deviations = quicksum(deviations[i] * deviations[i] for i in range(k))/k
    model.setObjective(avg_squared_deviations, GRB.MINIMIZE)

    model.addConstr(mu @ x >= b, "mean_return")
    model.addConstr(x.sum() == 1, "sum_to_one")  # Weights sum to 1
    model.setParam("Threads", 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([float(x[i].x) for i in range(m)])
        return x_opt
    else:
        print("No optimal solution found.")
        return None 


############# Network #############
def gurobi_network_first_stage(sample_k, s, C,Q_sp, Q_pc, R, M, H):
    # input:
    # C (unit cost for building a facility): p*1 numpy array
    # Q_sp (unit flow cost): s*p*g numpy array
    # Q_pc (unit flow cost): p*c*g numpy array
    # sample_S (supply): k*s*g numpy array
    # sample_D (demand): k*c*g numpy array
    # R (unit processing require): p*g numpy array
    # M (processing capacity): p*1 numpy array
    # H (multiplier): c*g numpy array
    # output:
    # x (open facility): p*1 numpy array
    # y_sp (flow supplier --> facility): s*p*g*k numpy array 
    # y_pc (flow facility --> consumer): p*c*g*k numpy array
    # z (multiplier): c*g*k numpy array
    sample_S = sample_k[:,:s,:]
    sample_D = sample_k[:,s:,:]
    _, p, g = Q_sp.shape
    k, c, _ = sample_D.shape
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
    
    model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sample_D[a, i, l]
                      for a in range(k) for l in range(g) for i in range(c)), name="demand")
    
    model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sample_S[a, i, l]
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



############# LP problem similar to maximum weight matching #############
def gurobi_LP(sample_k, N, w, A, seed = 1):
    # sample_k: k * 6 matrix, where each column corresponds the weight of an edge
    # N: number of nodes >= 4.
    # w: dictionary of edge weights, using 1-based index
    # A: some matrices of dimension N * N, used to destroy the total unimodularity, stored as a numpy matrix
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(1,4):
        for j in range(i+1, 5):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    model = Model("max_weight_matching")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam("Method", 0)
    # set seed
    model.setParam("Seed", seed)
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(w[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(A[i-1, j-1] * x[(min(i, j), max(i, j))] for j in range(1, N+1) if i != j) <= 1 for i in range(1, N+1))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[e].X for e in edges])
        return x_opt #, x.keys()
    else:
        print("No optimal solution found.")
        return None   



def gurobi_LP_full_random(sample_k, N, w, A, seed = 1, exact = False):
    # sample_k: k * N * (N-1)/2 matrix, where each column corresponds the weight of an edge
    # N: number of nodes >= 4.
    # w: dictionary of edge weights, using 1-based index
    # A: some matrices of dimension N * N, used to destroy the total unimodularity, stored as a numpy matrix
    weight = {}
    ind = 0
    if exact == True:
        for i in range(1,N):
            for j in range(i+1, N+1):
                weight[(i,j)] = w[(i,j)]
                ind += 1
    else:
        sample_mean = np.mean(sample_k, axis=0)
        for i in range(1,N):
            for j in range(i+1, N+1):
                weight[(i,j)] = sample_mean[ind]
                ind += 1
        
    model = Model("max_weight_matching")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam("Method", 0)
    # set seed
    model.setParam("Seed", seed)
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(weight[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(A[i-1, j-1] * x[(min(i, j), max(i, j))] for j in range(1, N+1) if i != j) <= 1 for i in range(1, N+1))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[e].X for e in edges])
        return x_opt #, x.keys()
    else:
        print("No optimal solution found.")
        return None 


def gurobi_LP_DRO_full_random(sample_k, N, w, A, seed = 1, varepsilon = 0):
    # sample_k: k * N * (N-1)/2 matrix, where each column corresponds the weight of an edge
    # N: number of nodes >= 4.
    # w: dictionary of edge weights, using 1-based index
    # A: some matrices of dimension N * N, used to destroy the total unimodularity, stored as a numpy matrix
    weight = {}
    ind = 0

    sample_mean = np.mean(sample_k, axis=0)
    for i in range(1,N):
        for j in range(i+1, N+1):
            weight[(i,j)] = sample_mean[ind] - varepsilon
            ind += 1
        
    model = Model("max_weight_matching_DRO")
    model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam("Method", 0)
    # set seed
    model.setParam("Seed", seed)
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    x = model.addVars(edges, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    
    model.setObjective(quicksum(weight[e] * x[e] for e in edges), GRB.MAXIMIZE)

    model.addConstrs(quicksum(A[i-1, j-1] * x[(min(i, j), max(i, j))] for j in range(1, N+1) if i != j) <= 1 for i in range(1, N+1))

    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[e].X for e in edges])
        return x_opt #, x.keys()
    else:
        print("No optimal solution found.")
        return None 


############# LASSO model selection problem #############
# note that for this problem, the only parameter is the beta_dict
def solve_model_selection(sample_k, beta_dict):
    # gurobi-like function to find the best beta value
    def get_sample_loss(sample_k, beta):
        # evaluate the MSE loss of a given beta based on samples 
        y = sample_k[:,0]
        X = sample_k[:,1:]
        return np.mean((y - np.dot(X, beta))**2)
    
    loss_dict = {}
    for lambda_val in beta_dict:
        obj = get_sample_loss(sample_k, beta_dict[lambda_val])
        loss_dict[lambda_val] = obj
    best_lambda = min(loss_dict, key = loss_dict.get)
    return best_lambda