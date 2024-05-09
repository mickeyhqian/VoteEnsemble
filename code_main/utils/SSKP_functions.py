from ParallelSolve import gurobi_SSKP, majority_vote, sequentialEvaluate, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit, gurobi_SSKP_DRO_wasserstein
import time
import numpy as np
from multiprocessing import Queue, Process
from utils.generateSamples import genSample_SSKP


# objective evaluation
def SSKP_eval(sample, x, r, c, q):
     # sample: large number * m matrix
     expected_cost = 0
     for w in sample:
          expected_cost += max(sum(w[i] * x[i] for i in range(len(x)))-q, 0)
     expected_cost = expected_cost/len(sample)
     opt = sum(r[i] * x[i] for i in range(len(x))) - c * expected_cost
     return opt

def SSKP_eval_wSol(sample, x, r, c, q):
     expected_cost = 0
     for w in sample:
          expected_cost += max(sum(w[i] * x[i] for i in range(len(x)))-q, 0)
     expected_cost = expected_cost/len(sample)
     opt = sum(r[i] * x[i] for i in range(len(x))) - c * expected_cost
     return opt, x

# functions that implements the comparison between SAA and Bagging-SAA, as well as evaluating the results
def comparison_final(B_list,k_list,number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # a unified function that compare SAA with different configurations of Bagging
    # prob_args: r,c,q
    # sample_args: paretoShapes
    # remark: r and paretoShapes automatically imply the number of products
    SAA_list = []
    bagging_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))] 
    for n in sample_number:
        SAA_intermediate = []
        bagging_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        for _ in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_SSKP, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {_}, SAA time: {time.time()-tic}")

            for ind1, B in enumerate(B_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging, _ = majority_vote(sample_n, B, int(n*k), gurobi_SSKP, rng_alg, *prob_args)
                    else:
                        bagging, _ = majority_vote(sample_n, B, k, gurobi_SSKP, rng_alg, *prob_args)
                    bagging_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {_}, B={B}, k={k}, Bagging time: {time.time()-tic}")
            
        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_list[ind1][ind2].append(bagging_intermediate[ind1][ind2])
    
    return SAA_list, bagging_list
        
def evaluation_final(SAA_list, bagging_list, large_number_sample, rng_sample, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_list), len(bagging_list[0])
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_list[ind1][ind2][i][j])
    
    solution_obj_values = {str(solution): 0 for solution in all_solutions}

    for solution in all_solutions:
        sample_large = genSample_SSKP(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
        obj_value = SSKP_eval(sample_large, solution, *prob_args)
        solution_obj_values[str(solution)] = obj_value
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_obj_list, bagging_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
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
                    bagging_obj = solution_obj_values[str(bagging_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
    
    return SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg

def evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    # parallel evaluation for SSKP (need modification for new problems)
    solution_obj_values = {str(solution): [] for solution in all_solutions}
    numProcesses = 9
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for solution in all_solutions:
        for i in range(eval_time):
            ######## places that need to change for different problems ########
            sample_large = genSample_SSKP(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
            ########
            taskLists[processIndex].append((sample_large, solution))
            processIndex = (processIndex + 1) % numProcesses
    
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            ######## places that need to change for different problems ########
            processList.append(Process(target = sequentialEvaluate, args = (SSKP_eval_wSol, queue, taskLists[i], *prob_args), daemon=True))
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



def evaluation_parallel(SAA_list, bagging_list, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_list), len(bagging_list[0])
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_list[ind1][ind2][i][j])
    
    solution_obj_values = evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args)

    SAA_obj_list, SAA_obj_avg = [], []
    bagging_obj_list, bagging_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
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
                    bagging_obj = solution_obj_values[str(bagging_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
    
    return SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg




################## functions used to evaluate Algorithms 3 and 4 ##################
# create a function that compares SAA, Algorithm 1, 3, and 4
# currently, suppose Algorithms 3 and 4 use the same k_list as Algorithm 1
# B12_list is a list of tuples, where each tuple is (B1, B2)
def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
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
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_SSKP, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind1, B in enumerate(B_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging, _ = majority_vote(sample_n, B, int(n*k), gurobi_SSKP, rng_alg, *prob_args)
                    else:
                        bagging, _ = majority_vote(sample_n, B, k, gurobi_SSKP, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")
            
            for ind1, (B1, B2) in enumerate(B12_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                        bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                    else:
                        bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                        bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
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


def SSKP_prob_comparison(B_list, k_list, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args):
    # function that compare the probability of outputting the optimal solution for SAA and Bagging-SAA (Algorithm 1)
    sample_large = genSample_SSKP(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
    SAA, _ = majority_vote(sample_large, 1, large_number_sample, gurobi_SSKP, rng_alg, *prob_args)

    x_star = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
    print("Optimal solution: ", x_star)

    SAA_prob_opt_list = [] # probability of outputting the optimal solution
    SAA_prob_diff_list = [] # difference between the probability of outputting the optimal solution and the largest probability of outputting a different solution
    SAA_prob_dist_list = [] # simulated distribution of the probability of outputting each solution
    bagging_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_prob_diff_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]

    for n in sample_number:
        # create dictionary to temporarily store the count for each sample size
        SAA_count = {}
        bagging_count = {}
        for B in B_list:
            for k in k_list:
                bagging_count[str((B,k))] = {}
        tic1 = time.time()
        for iter in range(number_of_iterations):
            tic2 = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_SSKP, rng_alg, *prob_args)
            SAA = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
            SAA_count[SAA] = SAA_count.get(SAA, 0) + 1/number_of_iterations
            for B in B_list:
                for k in k_list:
                    if k < 1:
                        bagging, _ = majority_vote(sample_n, B, int(n*k), gurobi_SSKP, rng_alg, *prob_args)
                    else:
                        bagging, _ = majority_vote(sample_n, B, k, gurobi_SSKP, rng_alg, *prob_args)
                    bagging = str(bagging) if type(bagging) == int else str(tuple(int(entry) for entry in bagging))
                    bagging_count[str((B,k))][bagging] = bagging_count[str((B,k))].get(bagging, 0) + 1/number_of_iterations
            print(f"Sample size {n}, iteration {iter}, time: {time.time()-tic2}")
        
        SAA_prob_opt_list.append(SAA_count.get(x_star, 0))
        max_prob = 0
        for key in SAA_count:
            if key != x_star:
                max_prob = max(max_prob, SAA_count[key])
        SAA_prob_diff_list.append(SAA_count.get(x_star, 0) - max_prob)
        SAA_prob_dist_list.append(SAA_count)

        for ind1, B in enumerate(B_list):
            for ind2, k in enumerate(k_list):
                bagging_prob_opt_list[ind1][ind2].append(bagging_count[str((B,k))].get(x_star, 0))
                max_prob = 0
                for key in bagging_count[str((B,k))]:
                    if key != x_star:
                        max_prob = max(max_prob, bagging_count[str((B,k))][key])
                bagging_prob_diff_list[ind1][ind2].append(bagging_count[str((B,k))].get(x_star, 0) - max_prob)
                bagging_prob_dist_list[ind1][ind2].append(bagging_count[str((B,k))])

        print(f"Sample size {n}, time: {time.time()-tic1}")
    
    return SAA_prob_opt_list, SAA_prob_diff_list, SAA_prob_dist_list, bagging_prob_opt_list, bagging_prob_diff_list, bagging_prob_dist_list, x_star

def comparison_DRO(B_list, k_list, B12_list, epsilon, tolerance, varepsilon_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    SAA_list = []
    dro_wasserstein_list = [[] for _ in range(len(varepsilon_list))]
    bagging_alg1_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    
    for n in sample_number:
        SAA_intermediate = []
        dro_wasserstein_intermediate = [[] for _ in range(len(varepsilon_list))]
        bagging_alg1_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_SSKP, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind, varepsilon in enumerate(varepsilon_list):
                tic = time.time()
                dro_wasserstein = gurobi_SSKP_DRO_wasserstein(sample_n, varepsilon, *prob_args)
                dro_wasserstein = tuple([round(x) for x in dro_wasserstein])
                dro_wasserstein_intermediate[ind].append(dro_wasserstein)
                print(f"Sample size {n}, iteration {iter}, varepsilon={varepsilon}, DRO time: {time.time()-tic}")
            
            for ind1, B in enumerate(B_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging, _ = majority_vote(sample_n, B, int(n*k), gurobi_SSKP, rng_alg, *prob_args)
                    else:
                        bagging, _ = majority_vote(sample_n, B, k, gurobi_SSKP, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")
            
            for ind1, (B1, B2) in enumerate(B12_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                        bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                    else:
                        bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                        bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_SSKP, SSKP_eval_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}")
            
        SAA_list.append(SAA_intermediate)

        for ind in range(len(varepsilon_list)):
            dro_wasserstein_list[ind].append(dro_wasserstein_intermediate[ind])
        
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_alg1_list[ind1][ind2].append(bagging_alg1_intermediate[ind1][ind2])
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
        
    return SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list



def evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    varepsilon_list_len = len(dro_wasserstein_list)
    B_list_len, k_list_len = len(bagging_alg1_list), len(bagging_alg1_list[0])
    B12_list_len = len(bagging_alg3_list)
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind in range(varepsilon_list_len):
                all_solutions.add(dro_wasserstein_list[ind][i][j])
            for ind2 in range(k_list_len):
                for ind1 in range(B_list_len):
                    all_solutions.add(bagging_alg1_list[ind1][ind2][i][j])
                for ind1 in range(B12_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])
    
    solution_obj_values = evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args)

    SAA_obj_list, SAA_obj_avg = [], []
    dro_wasserstein_obj_list, dro_wasserstein_obj_avg = [[] for _ in range(varepsilon_list_len)], [[] for _ in range(varepsilon_list_len)]
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

    for ind in range(varepsilon_list_len):
        for i in range(sample_number_len):
            current_dro_wasserstein_obj_list = []
            for j in range(number_of_iterations):
                dro_wasserstein_obj = solution_obj_values[str(dro_wasserstein_list[ind][i][j])]
                current_dro_wasserstein_obj_list.append(dro_wasserstein_obj)
            dro_wasserstein_obj_list[ind].append(current_dro_wasserstein_obj_list)
            dro_wasserstein_obj_avg[ind].append(np.mean(current_dro_wasserstein_obj_list))
    
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

    return SAA_obj_list, SAA_obj_avg, dro_wasserstein_obj_list, dro_wasserstein_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg

