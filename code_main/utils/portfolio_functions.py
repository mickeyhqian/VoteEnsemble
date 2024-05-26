import time
import numpy as np
from utils.generateSamples import genSample_portfolio
from ParallelSolve import gurobi_portfolio, majority_vote, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit

def portfolio_evalute_wSol(sample_n, x, p, mu, b, alpha):
    # portfolio evaluation function, basically computing the cVar directly.
    # p and alpha are problem parameters
    losses = - np.dot(sample_n, np.array(x) * np.array(p))
    idx = min(int(len(losses) * alpha), len(losses) - 1)
    var_threshold = np.quantile(losses, idx / (len(losses) - 1))
    cvar = np.sum(losses[losses > var_threshold] - var_threshold) / len(losses) / (1- alpha) + var_threshold
    return -cvar, x


def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # prob_args includes p, mu, b, alpha
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
            sample_n = genSample_portfolio(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_portfolio, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_portfolio, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, bagging1 time: {time.time()-tic}")
            
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
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

def evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, rng_sample, sample_args, *prob_args):
    # no need for parallel evaluation
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
        sample_large = genSample_portfolio(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
        obj_value, _ = portfolio_evalute_wSol(sample_large, solution, *prob_args)
        solution_obj_values[str(solution)] = obj_value
    
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
    


def portfolio_prob_comparison(B_list, k_list, B12_list, epsilon, tolerance, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args):
    sample_large = genSample_portfolio(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
    SAA, _ = majority_vote(sample_large, 1, large_number_sample, gurobi_portfolio, rng_alg, *prob_args)
    x_star = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
    print(f"Optimal solution: {x_star}")

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
            sample_n = genSample_portfolio(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_portfolio, rng_alg, *prob_args)
            SAA = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
            SAA_count[SAA] = SAA_count.get(SAA, 0) + 1/number_of_iterations
            
            for k in k_list:
                for B in B_list:
                    if k < 1:
                        baggingAlg1, _ = majority_vote(sample_n, B, int(n*k), gurobi_portfolio, rng_alg, *prob_args)
                    else:
                        baggingAlg1, _ = majority_vote(sample_n, B, k, gurobi_portfolio, rng_alg, *prob_args)
                    baggingAlg1 = str(baggingAlg1) if type(baggingAlg1) == int else str(tuple(int(entry) for entry in baggingAlg1))
                    baggingAlg1_count[str((B,k))][baggingAlg1] = baggingAlg1_count[str((B,k))].get(baggingAlg1, 0) + 1/number_of_iterations
                for B1, B2 in B12_list:
                    if k < 1:
                        baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                        baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                    else:
                        baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                        baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                    
                    baggingAlg3 = str(baggingAlg3) if type(baggingAlg3) == int else str(tuple(int(entry) for entry in baggingAlg3))
                    baggingAlg4 = str(baggingAlg4) if type(baggingAlg4) == int else str(tuple(int(entry) for entry in baggingAlg4))

                    baggingAlg3_count[str((B1,B2,k))][baggingAlg3] = baggingAlg3_count[str((B1,B2,k))].get(baggingAlg3, 0) + 1/number_of_iterations
                    baggingAlg4_count[str((B1,B2,k))][baggingAlg4] = baggingAlg4_count[str((B1,B2,k))].get(baggingAlg4, 0) + 1/number_of_iterations
            print(f"Sample size {n}, iteration {iter}, time: {time.time()-tic2}")

        SAA_prob_opt_list.append(SAA_count.get(x_star, 0))
        SAA_prob_dist_list.append(SAA_count)
        for ind2, k in enumerate(k_list):
            for ind1, B in enumerate(B_list):
                baggingAlg1_prob_opt_list[ind1][ind2].append(baggingAlg1_count[str((B,k))].get(x_star, 0))
                baggingAlg1_prob_dist_list[ind1][ind2].append(baggingAlg1_count[str((B,k))])
            for ind1, (B1,B2) in enumerate(B12_list):
                baggingAlg3_prob_opt_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))].get(x_star, 0))
                baggingAlg3_prob_dist_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))])
                baggingAlg4_prob_opt_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))].get(x_star, 0))
                baggingAlg4_prob_dist_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))])
        print(f"Sample size {n}, total time: {time.time()-tic1}")

    return SAA_prob_opt_list, SAA_prob_dist_list, baggingAlg1_prob_opt_list, baggingAlg1_prob_dist_list, baggingAlg3_prob_opt_list, baggingAlg3_prob_dist_list, baggingAlg4_prob_opt_list, baggingAlg4_prob_dist_list