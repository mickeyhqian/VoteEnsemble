import time
import numpy as np
from utils.generateSamples import genSample_SSKP
from ParallelSolve import majority_vote_LP, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, gurobi_LP_full_random, gurobi_LP_DRO_full_random

# in this version, the edge weight is entirely random, and w represents the mean of the edge weight

def get_pareto_params(N,w):
    pareto_params = []
    for i in range(1, N):
        for j in range(i+1, N+1):
            pareto_params.append(w[(i,j)]/(w[(i,j)]-1))
    return pareto_params


def check_exist(sol_set, x):
    if sol_set == set():
        return False
    for sol in sol_set:
        if np.allclose(sol, x, atol=1e-06):
            return True
    return False

def solution_retrieval(sample_args, N, w, A):
    # get enough solutions as the decision space
    rng = np.random.default_rng()
    sample_number = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    # sample_number = [50, 100, 200]
    B,k  = 400, 0.1
    retrieved_set = set()
    for _ in range(10): # repeat 10 times
        for n in sample_number:
            sample_n = genSample_SSKP(n, rng, type = sample_args['type'], params = sample_args['params'])
            _, retrieved_n = majority_vote_LP(sample_n, B, int(n*k), gurobi_LP_full_random, rng, N, w, A)
            for x in retrieved_n:
                x = tuple([float(item) for item in x])
                if not check_exist(retrieved_set, x):
                    retrieved_set.add(x)
    # ensure the optimal solution must in the set
    _, x_opt = LP_obj_optimal(N, w, A)
    if not check_exist(retrieved_set, x_opt):
        retrieved_set.add(tuple([float(item) for item in x_opt]))
    return retrieved_set

def solution_evaluation(retrieved_set, N, w, A):
    # input a set of solutions, output a dictionary of true objective values
    obj_values = {}
    for x in retrieved_set:
        obj_values[x] = LP_evaluate_exact(x, N, w, A)
    return obj_values

def solution_SAA_evaluation(retrieved_set, n, number_of_iterations, rng_sample, sample_args, N, w, A):
    # input a set of solutions, assuming they are the entire decision space
    # use SAA to evaluate the solutions, and return a dictionary of list of objective values
    x_count_dict = {iter: {x: None for x in retrieved_set} for iter in range(number_of_iterations)}
    obj_opt_dict = {iter: None for iter in range(number_of_iterations)}
    for iter in range(number_of_iterations):
        sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
        obj_opt = -np.inf
        for x in retrieved_set:
            obj_x, _ = LP_evaluate_wSol(sample_n, x, N, w, A)
            x_count_dict[iter][x] = obj_x
            obj_opt = max(obj_opt, obj_x)
        obj_opt_dict[iter] = obj_opt
    return x_count_dict, obj_opt_dict

def simulate_Alg34_pk(retrieved_set, x_count_dict, obj_opt_dict, epsilon):
    # compute p_k(x) for Alg3 and Alg4, which is the probability of including x in the suboptimal set
    pk_dict = {x: 0 for x in retrieved_set}
    for iter in x_count_dict:
        suboptimal_set = get_SAA_suboptimal_set(x_count_dict[iter], obj_opt_dict[iter], epsilon)
        for x in suboptimal_set:
            pk_dict[x] += 1/len(x_count_dict)
    return pk_dict        

def get_SAA_suboptimal_set(x_count,obj_opt,epsilon):
    # get SAA based suboptimal set
    # obj_opt is the SAA optimal objective value
    suboptimal_set = set()
    for x in x_count:
        if x_count[x] > obj_opt - epsilon:
            suboptimal_set.add(x)
    return suboptimal_set

def get_suboptimal_set(obj_values, obj_opt, delta):
    # get delta-suboptimal solutions based on obj_values dictionary
    # obj_opt = max(obj_values.values()) # assume the optimal solution is in the set
    suboptimal_set = set()
    for x in obj_values:
        if obj_values[x] > obj_opt - delta:
            suboptimal_set.add(x)
    return suboptimal_set

def simulate_SAA_pk(n, number_of_iterations, rng_sample, sample_args, N, w, A):
    pk_dict = {}
    retrieved_set = set()
    for _ in range(number_of_iterations):
        sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
        SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP_full_random, rng_sample, N, w, A)
        SAA = tuple([float(item) for item in SAA])
        if not check_exist(retrieved_set, SAA):
            retrieved_set.add(SAA)
            pk_dict[SAA] = 1/number_of_iterations
        else:
            pk_dict[SAA] += 1/number_of_iterations
    return pk_dict, retrieved_set

def check_SAA_optimal(pk_dict, retrieved_set, N, w ,A):
    obj_values = solution_evaluation(retrieved_set, N, w, A)
    obj_opt, _ = LP_obj_optimal(N, w, A)
    prob_opt = 0
    for x in pk_dict:
        if obj_values[x] >= obj_opt - 0.001:
            prob_opt += pk_dict[x]
    return prob_opt
        
def LP_eta_comparison(delta_list, epsilon, n, number_of_iterations, rng_sample, sample_args, N, w, A):
    # n: sample_number
    # for Alg1, just simulates p_k(x), and compute max p_k(x) - max_{x not in delta optimal set} p_k(x)
    # for Alg3 and Alg4, this p_k(x) is the probability to be included in the SAA suboptimal set
    pk_dict_SAA, _ = simulate_SAA_pk(n, number_of_iterations, rng_sample, sample_args, N, w, A)
    max_pk_SAA = max(pk_dict_SAA.values())
    
    retrieved_set = solution_retrieval(sample_args, N, w, A)
    x_count_dict, obj_opt_dict = solution_SAA_evaluation(retrieved_set, n, number_of_iterations, rng_sample, sample_args, N, w, A)
    pk_dict_Alg34 = simulate_Alg34_pk(retrieved_set, x_count_dict, obj_opt_dict, epsilon)
    max_pk_Alg34 = max(pk_dict_Alg34.values())

    obj_values = solution_evaluation(retrieved_set, N, w, A) # compute the true objective values
    obj_opt = max(obj_values.values())

    SAA_eta_list = []
    Alg34_eta_list = []
    for delta in delta_list:
        suboptimal_set_delta = get_suboptimal_set(obj_values,obj_opt, delta)
        # get the complement set: retrieved_set\suboptimal_set_delta
        not_suboptimal_set_delta = retrieved_set - suboptimal_set_delta
        max_pk_SAA_delta = 0 # the maximum probability of suboptimal solution
        for x in pk_dict_SAA:
            if check_exist(not_suboptimal_set_delta, x):
                max_pk_SAA_delta = max(max_pk_SAA_delta, pk_dict_SAA[x])

        max_pk_Alg34_delta = 0
        for x in not_suboptimal_set_delta:
            max_pk_Alg34_delta = max(max_pk_Alg34_delta, pk_dict_Alg34[x])
        
        SAA_eta_list.append(max_pk_SAA - max_pk_SAA_delta)
        Alg34_eta_list.append(max_pk_Alg34 - max_pk_Alg34_delta)

    return SAA_eta_list, Alg34_eta_list, retrieved_set, x_count_dict, obj_opt_dict, pk_dict_SAA, pk_dict_Alg34
        
    
# needs update
# def generateW(N, option = None):
    # # this script is also used to find good parameters
    # # using 1-based index
    # w = {}
    # for i in range(1, N):
    #     for j in range(i+1, N+1):
    #         if i <= 3 and j <= 4:
    #             w[(i,j)] = None # place holder
    #         elif N >= 8 and i >= N-3 and j >= N-2:
    #             if option == "random":
    #                 w[(i,j)] = np.random.uniform(1.95, 2.05)
    #             else:
    #                 w[(i,j)] = 2 # fixed weight
    #         else:
    #             if option == "random":
    #                 w[(i,j)] = np.random.uniform(1.95, 2.05)
    #             else:
    #                 w[(i,j)] = np.random.uniform(1.5, 2.5)
    # return w


############################################################################################################
def LP_obj_optimal(N, w, A, seed = None):
    # computes the optimal objective value
    if seed is not None:
        x_opt = gurobi_LP_full_random(None, N, w, A, seed = seed, exact = True)
    else:
        x_opt = gurobi_LP_full_random(None, N, w, A, exact = True)
    obj = LP_evaluate_exact(x_opt, N, w, A)
    return obj, x_opt
    

def LP_evaluate_exact(x, N, w, A):
    # one difference for this file is that no samples are needed.
    # x is the solution, represented as a tuple
    # note that, indices of x correspond to (1,2), (1,3), ..., (1,N), (2,3), ..., (N-1, N)
    edges = [(i, j) for i in range(1, N) for j in range(i + 1, N + 1)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1
    
    return obj

def LP_evaluate_wSol(sample_k, x, N, w, A):
    # sample-based evaluation
    ind, obj = 0, 0
    sample_mean = np.mean(sample_k, axis=0)
    
    for i in range(1, N):
        for j in range(i+1, N+1):
            obj += sample_mean[ind] * x[ind]
            ind += 1

    return obj, x

def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # assume k_list is a list of tuples, each tuple is (number, ratio), true k = max(number, ratio*n)
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
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP_full_random, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(item) for item in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind1, B in enumerate(B_list):
                for ind2, (num, ratio) in enumerate(k_list):
                    tic = time.time()
                    k = max(num, int(n*ratio))
                    bagging, _ = majority_vote_LP(sample_n, B, k, gurobi_LP_full_random, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

            for ind1, (B1, B2) in enumerate(B12_list):
                for ind2, (num, ratio) in enumerate(k_list):
                    tic = time.time()
                    k = max(num, int(n*ratio))
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
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
        solution_obj_values[str(solution)] = LP_evaluate_exact(solution, *prob_args)
    
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
        SAA_intermediate = []
        bagging_alg1_intermediate = []
        bagging_alg3_intermediate = [[] for _ in range(len(epsilon_list))]
        bagging_alg4_intermediate = [[] for _ in range(len(epsilon_list))]
        dyn_eps_alg3_intermediate = []
        dyn_eps_alg4_intermediate = []
        for iter in range(number_of_iterations):
            tic0 = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP_full_random, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(item) for item in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic0}")

            tic = time.time()
            k = max(num, int(n*ratio))
            bagging, _ = majority_vote_LP(sample_n, B, k, gurobi_LP_full_random, rng_alg, *prob_args)
            bagging_alg1_intermediate.append(tuple([float(item) for item in bagging]))
            print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

            for ind, epsilon in enumerate(epsilon_list):
                tic = time.time()
                bagging_alg3, _, _, eps_alg3 = baggingTwoPhase_woSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                bagging_alg4, _, _, eps_alg4 = baggingTwoPhase_wSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
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
        solution_obj_values[str(solution)] = LP_evaluate_exact(solution, *prob_args)
    
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
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP_full_random, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(item) for item in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            tic = time.time()
            for ind, varepsilon in enumerate(varepsilon_list):
                dro = gurobi_LP_DRO_full_random(sample_n, *prob_args, varepsilon=varepsilon)
                dro = tuple([float(item) for item in dro])
                dro_wasserstein_intermediate[ind].append(dro)
            print(f"Sample size {n}, iteration {iter}, DRO time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote_LP(sample_n, B, k, gurobi_LP_full_random, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")
                
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(item) for item in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4 time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
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
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg1_list[ind1][ind2][i][j])
            for ind1 in range(B12_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])
    
    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = LP_evaluate_exact(solution, *prob_args)

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
            


def tail_influence(B_list, k_tuple, B12_list, epsilon, tolerance, number_of_iterations, n, rng_sample, rng_alg, sample_args_list, prob_args_list):
    # simulate the probability of achieving the optimal solution for each problem instance
    # prob_args_list is a list of tuples/lists, corresponds to different problem instances
    SAA_prob_list = []
    bagging_alg1_prob_list = [[] for _ in range(len(B_list))]
    bagging_alg3_prob_list = [[] for _ in range(len(B12_list))]
    bagging_alg4_prob_list = [[] for _ in range(len(B12_list))]
    
    num, ratio = k_tuple
    k = max(num, int(n*ratio))
    for ind_instance, prob_args in enumerate(prob_args_list):
        SAA_opt_times = 0
        bagging_alg1_opt_times = [0 for _ in range(len(B_list))]
        bagging_alg3_opt_times = [0 for _ in range(len(B12_list))]
        bagging_alg4_opt_times = [0 for _ in range(len(B12_list))]
        
        obj_opt = LP_obj_optimal(*prob_args)[0]
        sample_args = sample_args_list[ind_instance]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_LP_full_random, rng_alg, *prob_args)
            if LP_evaluate_exact(SAA, *prob_args) >= obj_opt - 1e-6:
                SAA_opt_times += 1
            
            for ind, B in enumerate(B_list):
                bagging_alg1, _ = majority_vote_LP(sample_n, B, k, gurobi_LP_full_random, rng_alg, *prob_args)
                if LP_evaluate_exact(bagging_alg1, *prob_args) >= obj_opt - 1e-6:
                    bagging_alg1_opt_times[ind] += 1
            
            for ind, B12 in enumerate(B12_list):
                bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                if LP_evaluate_exact(bagging_alg3, *prob_args) >= obj_opt - 1e-6:
                    bagging_alg3_opt_times[ind] += 1
                print(f"Problem instance {ind_instance}, iteration {iter}, B12={B12}, adaptive epsilon: {eps_dyn}")
                
                bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_LP_full_random, LP_evaluate_wSol, rng_alg, *prob_args)
                if LP_evaluate_exact(bagging_alg4, *prob_args) >= obj_opt - 1e-6:
                    bagging_alg4_opt_times[ind] += 1
            
            print(f"Problem instance {ind_instance}, iteration {iter}, total time: {time.time()-tic}")

        SAA_prob_list.append(SAA_opt_times/number_of_iterations)
        for ind in range(len(B_list)):
            bagging_alg1_prob_list[ind].append(bagging_alg1_opt_times[ind]/number_of_iterations)
        for ind in range(len(B12_list)):
            bagging_alg3_prob_list[ind].append(bagging_alg3_opt_times[ind]/number_of_iterations)
            bagging_alg4_prob_list[ind].append(bagging_alg4_opt_times[ind]/number_of_iterations)

    return SAA_prob_list, bagging_alg1_prob_list, bagging_alg3_prob_list, bagging_alg4_prob_list
