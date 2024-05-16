import time
import numpy as np
from utils.generateSamples import genSample_portfolio
from ParallelSolve import gurobi_portfolio_continuous, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, majority_vote_LP

# min variance of current portfolio = 
# sum 1/n \cdot \sum_i=1^n (\xi_i ^T x - \mu^T x)^2
# s.t. mu^T x >= b
# sum(x) = 1
# x >= 0

def get_pareto_params(mu):
    return [item/(item-1) for item in mu]

def portfolio_evalute_wSol(sample_k, x, mu, b):
    # sample-based portfolio evaluation function
    # k,m = sample_k.shape
    mu = np.array(mu).reshape(1,-1)
    x = np.array(x).reshape(-1,1)
    sample_k_adjusted = sample_k - mu
    squared_deviations = np.dot(sample_k_adjusted, x) ** 2
    return - np.mean(squared_deviations), x
    

# E[(\xi^T x - \mu^T x)^2] = E[(\xi^T x )^2] - (\mu^T x)^2 = x^T * E[\xi \xi^T] * x - (mu^T x)^2
def portfolio_evaluate_exact(x,mu,b):
    # exact portfolio evaluation function
    # assuming the sample is pareto-distributed, so we can get the shape parameters from mu
    # assume that the covariance matrix is diagonal
    shapes = [item/(item-1) for item in mu]
    variance = [shape/((shape-1)**2 * (shape-2)) for shape in shapes]
    # print(variance, x)
    # print(x, x.shape, cov_matrix, cov_matrix.shape)
    # print(- np.dot(np.dot(x.T, cov_matrix), x), type(- np.dot(np.dot(x.T, cov_matrix), x)))
    return sum([variance[i] * x[i]**2 for i in range(len(x))])
    

def comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # prob_args includes p, mu, b, alpha
    SAA_list = []
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_portfolio(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_portfolio_continuous, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
            
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio_continuous, portfolio_evalute_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio_continuous, portfolio_evalute_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
        
        SAA_list.append(SAA_intermediate)
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg3_list, bagging_alg4_list

def evaluation_twoPhase(SAA_list, bagging_alg3_list, bagging_alg4_list, *prob_args):
    # no need for parallel evaluation
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    k_list_len = len(bagging_alg3_list[0])
    B12_list_len = len(bagging_alg3_list)
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]

    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = portfolio_evaluate_exact(SAA_list[i][j], *prob_args)
            current_SAA_obj_list.append(SAA_obj)
            print("Type:" , type(SAA_obj))
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))
            
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = portfolio_evaluate_exact(bagging_alg3_list[ind1][ind2][i][j], *prob_args)
                    bagging_obj_4 = portfolio_evaluate_exact(bagging_alg4_list[ind1][ind2][i][j], *prob_args)
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg
    

# def portfolio_prob_comparison(B_list, k_list, B12_list, epsilon, tolerance, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args):
#     sample_large = genSample_portfolio(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
#     SAA, _ = majority_vote(sample_large, 1, large_number_sample, gurobi_portfolio, rng_alg, *prob_args)
#     x_star = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
#     print(f"Optimal solution: {x_star}")

#     SAA_prob_opt_list = []
#     SAA_prob_dist_list = []
#     baggingAlg1_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
#     baggingAlg1_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
#     baggingAlg3_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
#     baggingAlg3_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
#     baggingAlg4_prob_opt_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
#     baggingAlg4_prob_dist_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

#     for n in sample_number:
#         SAA_count = {}
#         baggingAlg1_count = {}
#         baggingAlg3_count = {}
#         baggingAlg4_count = {}
#         for k in k_list:
#             for B in B_list:
#                 baggingAlg1_count[str((B,k))] = {}
#             for B1, B2 in B12_list:
#                 baggingAlg3_count[str((B1,B2,k))] = {}
#                 baggingAlg4_count[str((B1,B2,k))] = {}
        
#         tic1 = time.time()
#         for iter in range(number_of_iterations):
#             tic2 = time.time()
#             sample_n = genSample_portfolio(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
#             SAA, _ = majority_vote(sample_n, 1, n, gurobi_portfolio, rng_alg, *prob_args)
#             SAA = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
#             SAA_count[SAA] = SAA_count.get(SAA, 0) + 1/number_of_iterations
            
#             for k in k_list:
#                 for B in B_list:
#                     if k < 1:
#                         baggingAlg1, _ = majority_vote(sample_n, B, int(n*k), gurobi_portfolio, rng_alg, *prob_args)
#                     else:
#                         baggingAlg1, _ = majority_vote(sample_n, B, k, gurobi_portfolio, rng_alg, *prob_args)
#                     baggingAlg1 = str(baggingAlg1) if type(baggingAlg1) == int else str(tuple(int(entry) for entry in baggingAlg1))
#                     baggingAlg1_count[str((B,k))][baggingAlg1] = baggingAlg1_count[str((B,k))].get(baggingAlg1, 0) + 1/number_of_iterations
#                 for B1, B2 in B12_list:
#                     if k < 1:
#                         baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
#                         baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, int(n*k), epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
#                     else:
#                         baggingAlg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
#                         baggingAlg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio, portfolio_evalute_wSol, rng_alg, *prob_args)
                    
#                     baggingAlg3 = str(baggingAlg3) if type(baggingAlg3) == int else str(tuple(int(entry) for entry in baggingAlg3))
#                     baggingAlg4 = str(baggingAlg4) if type(baggingAlg4) == int else str(tuple(int(entry) for entry in baggingAlg4))

#                     baggingAlg3_count[str((B1,B2,k))][baggingAlg3] = baggingAlg3_count[str((B1,B2,k))].get(baggingAlg3, 0) + 1/number_of_iterations
#                     baggingAlg4_count[str((B1,B2,k))][baggingAlg4] = baggingAlg4_count[str((B1,B2,k))].get(baggingAlg4, 0) + 1/number_of_iterations
#             print(f"Sample size {n}, iteration {iter}, time: {time.time()-tic2}")

#         SAA_prob_opt_list.append(SAA_count.get(x_star, 0))
#         SAA_prob_dist_list.append(SAA_count)
#         for ind2, k in enumerate(k_list):
#             for ind1, B in enumerate(B_list):
#                 baggingAlg1_prob_opt_list[ind1][ind2].append(baggingAlg1_count[str((B,k))].get(x_star, 0))
#                 baggingAlg1_prob_dist_list[ind1][ind2].append(baggingAlg1_count[str((B,k))])
#             for ind1, (B1,B2) in enumerate(B12_list):
#                 baggingAlg3_prob_opt_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))].get(x_star, 0))
#                 baggingAlg3_prob_dist_list[ind1][ind2].append(baggingAlg3_count[str((B1,B2,k))])
#                 baggingAlg4_prob_opt_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))].get(x_star, 0))
#                 baggingAlg4_prob_dist_list[ind1][ind2].append(baggingAlg4_count[str((B1,B2,k))])
#         print(f"Sample size {n}, total time: {time.time()-tic1}")

#     return SAA_prob_opt_list, SAA_prob_dist_list, baggingAlg1_prob_opt_list, baggingAlg1_prob_dist_list, baggingAlg3_prob_opt_list, baggingAlg3_prob_dist_list, baggingAlg4_prob_opt_list, baggingAlg4_prob_dist_list