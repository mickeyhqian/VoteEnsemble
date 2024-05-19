import time
import numpy as np
from utils.generateSamples import genSample_SSKP
from ParallelSolve import majority_vote, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit, gurobi_matching, gurobi_matching_DRO_wasserstein

# None: already modified to the bipartite version.

def generateW(N, option = None):
    # generate the weight matrix for the maximum weight bipartite matching problem
    # this script is also used to find good parameters
    # using 0-based index, need N >= 6
    w = {}
    for i in range(N):
        for j in range(N):
            if i < 3 and j < 3:
                w[(i,j)] = None
            elif i >= N-2 and j >= N-2:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    # w[(i,j)] = 2
                    w[(i,j)] = np.random.uniform(1.9, 2.1)
            else:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    # w[(i,j)] = np.random.uniform(1.9, 2)
                    w[(i,j)] = 2.1
    return w

def matching_obj_optimal(sample_args, N, w):
    # computes the optimal objective value
    if sample_args['type'] == 'pareto' or sample_args['type'] == 'sym_pareto' or sample_args['type'] == 'neg_pareto':
        sample_mean = np.reshape([item/(item-1) for item in sample_args['params']], (1, len(sample_args['params'])))
    elif sample_args['type'] == 'normal':
        sample_mean = np.reshape(sample_args['params'][0], (1, len(sample_args['params'][0])))
    x_opt = gurobi_matching(sample_mean, N, w)
    obj = matching_evaluate_exact(sample_args, x_opt, N, w)
    return obj, x_opt
    

def matching_evaluate_exact(sample_args, x, N, w):
    # evaluate the objective value of a given solution
    # x is the solution, represented as a tuple
    # first, retrieve the sample mean and fill the None values in w
    if sample_args['type'] == 'pareto' or sample_args['type'] == 'sym_pareto' or sample_args['type'] == 'neg_pareto':
        sample_mean = [item/(item-1) for item in sample_args['params']]
    elif sample_args['type'] == 'normal':
        sample_mean = sample_args['params'][0]
    
    ind = 0
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    # note that, indices of x correspond to (0,0), (0,2), ..., (0,N-1), (1,0), (1,1), ..., (N-1,N-1)
    edges = [(i, j) for i in range(N) for j in range(N)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1

    return obj

def matching_evaluate_wSol(sample_k, x, N, w):
    # the sample-based version of matching_evaluate_exact function, also returns the solution
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    edges = [(i, j) for i in range(N) for j in range(N)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1

    return obj, x

def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # for maximum weight matching problem, prob_args represent N and w.
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
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num,ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
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

def evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, *prob_args):
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

    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
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


# TODO: need to modify based on the new version of k_tuple
# determine if a solution is optimal based on objective value
def matching_prob_comparison(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, *prob_args):
    # function that compare the probability of outputting the optimal solution for SAA and three variants of Bagging-SAA
    # first, solve the optimal objective value
    obj_opt, x_opt = matching_obj_optimal(sample_args, *prob_args)
    opt_set = set(tuple([round(x) for x in x_opt]))
    subopt_set = set()
    print(f"Optimal objective value: {obj_opt}")

    # initialize lists to store the results
    # the opt_list stores the probability of outputting the optimal solution
    # the dist_list stores the probability distribution of the solutions
    SAA_prob_opt_list = []
    SAA_prob_dist_list = []
    baggingAlg1_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    baggingAlg1_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    baggingAlg3_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    baggingAlg3_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    baggingAlg4_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    baggingAlg4_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_count = {}
        baggingAlg1_count = {}
        baggingAlg3_count = {}
        baggingAlg4_count = {}
        for k in k_list:
            for B in B_list:
                baggingAlg1_count[str((B,k))] = {}
            for B1, B2 in B12_list:
                baggingAlg3_count[str((B1,B2,k))] = {}
                baggingAlg4_count[str((B1,B2,k))] = {}
        
        tic1 = time.time()
        for iter in range(number_of_iterations):
            tic2 = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            if SAA in opt_set:
                SAA_count['opt'] = SAA_count.get('opt', 0) + 1/number_of_iterations
            elif SAA not in subopt_set:
                SAA_obj = matching_evaluate_exact(sample_args, SAA, *prob_args)
                if abs(SAA_obj - obj_opt) < 1e-6:
                    SAA_count['opt'] = SAA_count.get('opt', 0) + 1/number_of_iterations
                    opt_set.add(SAA)
                else:
                    subopt_set.add(SAA)
            SAA = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
            SAA_count[SAA] = SAA_count.get(SAA, 0) + 1/number_of_iterations
            
            for k in k_list:
                for B in B_list:
                    if k < 1:
                        baggingAlg1, _ = majority_vote(sample_n, B, int(n*k), gurobi_matching, rng_alg, *prob_args)
                    else:
                        baggingAlg1, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
                    if baggingAlg1 in opt_set:
                        baggingAlg1_count[str((B,k))]['opt'] = baggingAlg1_count[str((B,k))].get('opt', 0) + 1/number_of_iterations
                    elif baggingAlg1 not in subopt_set:
                        baggingAlg1_obj = matching_evaluate_exact(sample_args, baggingAlg1, *prob_args)
                        if abs(baggingAlg1_obj - obj_opt) < 1e-6:
                            baggingAlg1_count[str((B,k))]['opt'] = baggingAlg1_count[str((B,k))].get('opt', 0) + 1/number_of_iterations
                            opt_set.add(baggingAlg1)
                        else:
                            subopt_set.add(baggingAlg1)
                    baggingAlg1 = str(baggingAlg1) if type(baggingAlg1) == int else str(tuple(int(entry) for entry in baggingAlg1))
                    baggingAlg1_count[str((B,k))][baggingAlg1] = baggingAlg1_count[str((B,k))].get(baggingAlg1, 0) + 1/number_of_iterations
                
                for B1, B2 in B12_list:
                    if k < 1:
                        baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                        baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    else:
                        baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                        baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    
                    if baggingAlg3 in opt_set:
                        baggingAlg3_count[str((B1,B2,k))]['opt'] = baggingAlg3_count[str((B1,B2,k))].get('opt', 0) + 1/number_of_iterations
                    elif baggingAlg3 not in subopt_set:
                        baggingAlg3_obj = matching_evaluate_exact(sample_args, baggingAlg3, *prob_args)
                        if abs(baggingAlg3_obj - obj_opt) < 1e-6:
                            baggingAlg3_count[str((B1,B2,k))]['opt'] = baggingAlg3_count[str((B1,B2,k))].get('opt', 0) + 1/number_of_iterations
                            opt_set.add(baggingAlg3)
                        else:
                            subopt_set.add(baggingAlg3)
                    
                    if baggingAlg4 in opt_set:
                        baggingAlg4_count[str((B1,B2,k))]['opt'] = baggingAlg4_count[str((B1,B2,k))].get('opt', 0) + 1/number_of_iterations
                    elif baggingAlg4 not in subopt_set:
                        baggingAlg4_obj = matching_evaluate_exact(sample_args, baggingAlg4, *prob_args)
                        if abs(baggingAlg4_obj - obj_opt) < 1e-6:
                            baggingAlg4_count[str((B1,B2,k))]['opt'] = baggingAlg4_count[str((B1,B2,k))].get('opt', 0) + 1/number_of_iterations
                            opt_set.add(baggingAlg4)
                        else:
                            subopt_set.add(baggingAlg4)
                    
                    baggingAlg3 = str(baggingAlg3) if type(baggingAlg3) == int else str(tuple(int(entry) for entry in baggingAlg3))
                    baggingAlg4 = str(baggingAlg4) if type(baggingAlg4) == int else str(tuple(int(entry) for entry in baggingAlg4))
                    baggingAlg3_count[str((B1,B2,k))][baggingAlg3] = baggingAlg3_count[str((B1,B2,k))].get(baggingAlg3, 0) + 1/number_of_iterations
                    baggingAlg4_count[str((B1,B2,k))][baggingAlg4] = baggingAlg4_count[str((B1,B2,k))].get(baggingAlg4, 0) + 1/number_of_iterations

            print(f"Sample size {n}, iteration {iter}, time: {time.time()-tic2}")
        
        SAA_prob_opt_list.append(SAA_count.get('opt', 0))
        SAA_prob_dist_list.append(SAA_count)

        for ind2, k in enumerate(k_list):
            for ind1, B in enumerate(B_list):
                baggingAlg1_prob_opt_list[ind1][ind2].append(baggingAlg1_count[str((B,k))].get('opt', 0))
                baggingAlg1_prob_dist_list[ind1][ind2].append(baggingAlg1_count[str((B,k))])
            
            for ind1, (B1, B2) in enumerate(B12_list):
                baggingAlg3_prob_opt_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))].get('opt', 0))
                baggingAlg3_prob_dist_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))])
                baggingAlg4_prob_opt_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))].get('opt', 0))
                baggingAlg4_prob_dist_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))])
        
        print(f"Sample size {n}, total time: {time.time()-tic1}")

    return SAA_prob_opt_list, SAA_prob_dist_list, baggingAlg1_prob_opt_list, baggingAlg1_prob_dist_list, baggingAlg3_prob_opt_list, baggingAlg3_prob_dist_list, baggingAlg4_prob_opt_list, baggingAlg4_prob_dist_list

                        
            

def comparison_epsilon(B, k, B12, epsilon_list, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, *prob_args):
    # function that performs grid-search on the values of epsilon. Only consider one possible value for B, k, and B12.
    SAA_list = []
    bagging_alg1_list = []
    bagging_alg3_list = [[] for _ in range(len(epsilon_list))]
    bagging_alg4_list = [[] for _ in range(len(epsilon_list))]
    # use lists to store the dynamic epsilon values, each entry corresponds to the list of epsilon values for a given sample size
    # under a bunch of repeated experiments
    dyn_eps_alg3_list = []
    dyn_eps_alg4_list = []

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg1_intermediate = []
        bagging_alg3_intermediate = [[] for _ in range(len(epsilon_list))]
        bagging_alg4_intermediate = [[] for _ in range(len(epsilon_list))]
        dyn_eps_alg3_intermediate = []
        dyn_eps_alg4_intermediate = []
        for iter in range(number_of_iterations):
            tic0 = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic0}")

            tic = time.time()
            if k < 1:
                bagging, _ = majority_vote(sample_n, B, int(n*k), gurobi_matching, rng_alg, *prob_args)
            else:
                bagging, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
            bagging_alg1_intermediate.append(tuple([round(x) for x in bagging]))
            print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

            for ind, epsilon in enumerate(epsilon_list):
                tic = time.time()
                if k < 1:
                    bagging_alg3, _, _, eps_alg3 = baggingTwoPhase_woSplit(sample_n, B12[0], B12[1], int(n*k), epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, eps_alg4 = baggingTwoPhase_wSplit(sample_n, B12[0], B12[1], int(n*k), epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                else:
                    bagging_alg3, _, _, eps_alg3 = baggingTwoPhase_woSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, eps_alg4 = baggingTwoPhase_wSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                bagging_alg3_intermediate[ind].append(tuple([round(x) for x in bagging_alg3]))
                bagging_alg4_intermediate[ind].append(tuple([round(x) for x in bagging_alg4]))
                if epsilon == "dynamic":
                    dyn_eps_alg3_intermediate.append(eps_alg3)
                    dyn_eps_alg4_intermediate.append(eps_alg4)
                print(f"Sample size {n}, iteration {iter}, B1={B12[0]}, B2={B12[1]}, epsilon={epsilon}, Bagging Alg 3 & 4 time: {time.time()-tic}")
            
            print(f"Sample size {n}, iteration {iter}, total time: {time.time()-tic0}")
            
        SAA_list.append(SAA_intermediate)
        bagging_alg1_list.append(bagging_alg1_intermediate)
        dyn_eps_alg3_list.append(dyn_eps_alg3_intermediate)
        dyn_eps_alg4_list.append(dyn_eps_alg4_intermediate)
        for ind in range(len(epsilon_list)):
            bagging_alg3_list[ind].append(bagging_alg3_intermediate[ind])
            bagging_alg4_list[ind].append(bagging_alg4_intermediate[ind])

    return SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, dyn_eps_alg3_list, dyn_eps_alg4_list

def evaluation_epsilon(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, *prob_args):
    # evaluation function for the epsilon comparison
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    epsilon_list_len = len(bagging_alg3_list)
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            all_solutions.add(bagging_alg1_list[i][j])
            for ind in range(epsilon_list_len):
                all_solutions.add(bagging_alg3_list[ind][i][j])
                all_solutions.add(bagging_alg4_list[ind][i][j])
    
    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg1_obj_list, bagging_alg1_obj_avg = [], []
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[] for _ in range(epsilon_list_len)], [[] for _ in range(epsilon_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[] for _ in range(epsilon_list_len)], [[] for _ in range(epsilon_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        current_bagging_alg1_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
            bagging_alg1_obj = solution_obj_values[str(bagging_alg1_list[i][j])]
            current_bagging_alg1_obj_list.append(bagging_alg1_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))
        bagging_alg1_obj_list.append(current_bagging_alg1_obj_list)
        bagging_alg1_obj_avg.append(np.mean(current_bagging_alg1_obj_list))
    
    for ind in range(epsilon_list_len):
        for i in range(sample_number_len):
            current_bagging_alg3_obj_list = []
            current_bagging_alg4_obj_list = []
            for j in range(number_of_iterations):
                bagging_alg3_obj = solution_obj_values[str(bagging_alg3_list[ind][i][j])]
                current_bagging_alg3_obj_list.append(bagging_alg3_obj)
                bagging_alg4_obj = solution_obj_values[str(bagging_alg4_list[ind][i][j])]
                current_bagging_alg4_obj_list.append(bagging_alg4_obj)
            bagging_alg3_obj_list[ind].append(current_bagging_alg3_obj_list)
            bagging_alg3_obj_avg[ind].append(np.mean(current_bagging_alg3_obj_list))
            bagging_alg4_obj_list[ind].append(current_bagging_alg4_obj_list)
            bagging_alg4_obj_avg[ind].append(np.mean(current_bagging_alg4_obj_list))
    
    return SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg


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
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind, varepsilon in enumerate(varepsilon_list):
                tic = time.time()
                dro_wasserstein = gurobi_matching_DRO_wasserstein(sample_n, *prob_args, varepsilon= varepsilon)
                dro_wasserstein = tuple([round(x) for x in dro_wasserstein])
                dro_wasserstein_intermediate[ind].append(dro_wasserstein)
                print(f"Sample size {n}, iteration {iter}, varepsilon={varepsilon}, DRO time: {time.time()-tic}")
            
            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
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

def evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, *prob_args):
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

    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
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