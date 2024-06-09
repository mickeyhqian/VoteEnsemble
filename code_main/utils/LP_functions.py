import time
import numpy as np
from utils.generateSamples import genSample_SSKP
from ParallelSolve import majority_vote_LP, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, gurobi_LP


def generateW(N, option = None):
    # this script is also used to find good parameters
    # using 1-based index
    w = {}
    for i in range(1, N):
        for j in range(i+1, N+1):
            if i <= 3 and j <= 4:
                w[(i,j)] = None # place holder
            elif N >= 8 and i >= N-3 and j >= N-2:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    w[(i,j)] = 2 # fixed weight
            else:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    w[(i,j)] = np.random.uniform(1.5, 2.5)
    return w

def LP_obj_optimal(sample_args, N, w, A, seed = None):
    # computes the optimal objective value
    if sample_args['type'] == 'pareto':
        sample_mean = np.reshape([item/(item-1) for item in sample_args['params']], (1, len(sample_args['params'])))
    elif sample_args['type'] == 'normal':
        sample_mean = np.reshape(sample_args['params'][0], (1, len(sample_args['params'][0])))
    if seed is not None:
        x_opt = gurobi_LP(sample_mean, N, w, A, seed = seed)    
    else:
        x_opt = gurobi_LP(sample_mean, N, w, A)
    obj = LP_evaluate_exact(sample_args, x_opt, N, w, A)
    return obj, x_opt
    

def LP_evaluate_exact(sample_args, x, N, w, A):
    # one difference for this file is that no samples are needed.
    # x is the solution, represented as a tuple
    # first, retrieve the sample mean and fill the None values in w
    ind = 0
    if sample_args['type'] == 'pareto':
        sample_mean = [item/(item-1) for item in sample_args['params']]
    elif sample_args['type'] == 'normal':
        sample_mean = sample_args['params'][0]
    for i in range(1,4):
        for j in range(i+1, 5):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    # note that, indices of x correspond to (1,2), (1,3), ..., (1,N), (2,3), ..., (N-1, N)
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1
    
    return obj

def LP_evaluate_wSol(sample_k, x, N, w,A):
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(1,4):
        for j in range(i+1, 5):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1

    return obj, x

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
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(item) for item in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote_LP(sample_n, B, k, gurobi_LP, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg4]))
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
        solution_obj_values[str(solution)] = LP_evaluate_exact(sample_args, solution, *prob_args)
    
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




def comparison_epsilon(B, k_tuple, B12, epsilon_list, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, *prob_args):
    SAA_list = []
    bagging_alg1_list = []
    bagging_alg3_list = [[] for _ in range(len(epsilon_list))]
    bagging_alg4_list = [[] for _ in range(len(epsilon_list))]
    dyn_eps_alg3_list = []
    dyn_eps_alg4_list = []

    num, ratio = k_tuple
    for n in sample_number:
        k = max(num, int(n*ratio))
        SAA_intermediate = []
        bagging_alg1_intermediate = []
        bagging_alg3_intermediate = [[] for _ in range(len(epsilon_list))]
        bagging_alg4_intermediate = [[] for _ in range(len(epsilon_list))]
        dyn_eps_alg3_intermediate = []
        dyn_eps_alg4_intermediate = []
        for iter in range(number_of_iterations):
            tic0 = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(item) for item in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic0}")

            tic = time.time()
            bagging, _ = majority_vote_LP(sample_n, B, k, gurobi_LP, rng_alg, *prob_args)
            bagging_alg1_intermediate.append(tuple([float(item) for item in bagging]))
            print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

            for ind, epsilon in enumerate(epsilon_list):
                tic = time.time()
                bagging_alg3, _, _, eps_alg3 = baggingTwoPhase_woSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP, LP_evaluate_wSol, rng_alg, *prob_args)
                bagging_alg4, _, _, eps_alg4 = baggingTwoPhase_wSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP, LP_evaluate_wSol, rng_alg, *prob_args)
                bagging_alg3_intermediate[ind].append(tuple([float(item) for item in bagging_alg3]))
                bagging_alg4_intermediate[ind].append(tuple([float(item) for item in bagging_alg4]))
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
        solution_obj_values[str(solution)] = LP_evaluate_exact(sample_args, solution, *prob_args)
    
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
    