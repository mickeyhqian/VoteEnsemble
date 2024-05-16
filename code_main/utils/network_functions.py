from gurobipy import Model, GRB, quicksum
import numpy as np
import time
import json
from utils.generateSamples import genSample_network
from ParallelSolve import gurobi_network_first_stage, majority_vote, sequentialEvaluate, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit
from multiprocessing import Queue, Process



def gurobi_second_stage_wSol(sample_n, x, s, C, Q_sp, Q_pc, R, M, H):
    # second stage LP problem
    # tic = time.time()
    sample_S = sample_n[:,:s,:]
    sample_D = sample_n[:,s:,:]
    _, p, g = Q_sp.shape
    n, c, _ = sample_D.shape
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
    
    model.addConstrs((quicksum(y_pc[j, i, l, a] + z[i, l, a] for j in range(p)) >= sample_D[a, i, l]
                        for a in range(n) for l in range(g) for i in range(c)), name="demand")
    
    model.addConstrs((quicksum(y_sp[i, j, l, a] for j in range(p)) <= sample_S[a, i, l]
                        for a in range(n) for l in range(g) for i in range(s)), name="supply")
    
    model.addConstrs((quicksum(R[j, l] * quicksum(y_sp[i, j, l, a] for i in range(s)) for l in range(g)) <= M[j] * x[j]
                        for a in range(n) for j in range(p)), name="capacity")
    
    model.setParam("Threads", 9)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # print(f"Time elapsed: {time.time()-tic}")
        # return model.ObjVal + sum(C[j] * x[j] for j in range(p)), x
        return -model.ObjVal - sum(C[j] * x[j] for j in range(p)), x # turn the problem into a maximization problem
    else:
        print("No optimal solution found.")
        return None
    
def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # prob_args include s, C, Q_sp, Q_pc, R, M, H
    SAA_list = []
    bagging_alg1_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg1_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_network(n, rng_sample, type=sample_args["type"], size = sample_args["size"], params=sample_args["params"])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_network_first_stage, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))    
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_network_first_stage, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")
            
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_network_first_stage, gurobi_second_stage_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_network_first_stage, gurobi_second_stage_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}")
            
        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_alg1_list[ind1][ind2].append(bagging_alg1_intermediate[ind1][ind2])
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list


def evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_alg1_list), len(bagging_alg1_list[0])
    B12_list_len = len(bagging_alg3_list)
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg1_list[ind1][ind2][i][j])
            for ind1 in range(B12_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])

    solution_obj_values = evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args)

    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg1_obj_list, bagging_alg1_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list = []
                for j in range(number_of_iterations):
                    bagging_obj = solution_obj_values[str(bagging_alg1_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_alg1_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_alg1_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
            
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = solution_obj_values[str(bagging_alg3_list[ind1][ind2][i][j])]
                    bagging_obj_4 = solution_obj_values[str(bagging_alg4_list[ind1][ind2][i][j])]
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg


def evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    # parallel evaluation for SSKP (need modification for new problems)
    solution_obj_values = {str(solution): [] for solution in all_solutions}
    numProcesses = 1
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for solution in all_solutions:
        for i in range(eval_time):
            ######## places that need to change for different problems ########
            sample_large = genSample_network(large_number_sample, rng_sample, type=sample_args["type"], size = sample_args["size"], params=sample_args["params"])
            ########
            taskLists[processIndex].append((sample_large, solution))
            processIndex = (processIndex + 1) % numProcesses
    
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            ######## places that need to change for different problems ########
            processList.append(Process(target = sequentialEvaluate, args = (gurobi_second_stage_wSol, queue, taskLists[i], *prob_args), daemon=True))
            ########

    for process in processList:
        process.start()
    
    for _ in range(len(all_solutions) * eval_time):
        obj_value, solution = queue.get()
        if solution is not None:
            solution_obj_values[str(solution)].append(obj_value)
    
    for process in processList:
        process.join() 
    
    solution_obj_values = {key: np.mean(value) for key, value in solution_obj_values.items()}

    return solution_obj_values