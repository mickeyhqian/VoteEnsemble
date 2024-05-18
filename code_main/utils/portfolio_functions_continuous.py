import time
import numpy as np
from utils.generateSamples import genSample_portfolio
from ParallelSolve import gurobi_portfolio_continuous, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, majority_vote_LP

from ParallelSolve import test_baggingTwoPhase_woSplit_LP

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
    return - np.mean(squared_deviations), x # use the negative value for using Algorithms 3 and 4
    

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
    

def comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args, k2_tuple = None):
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
                if k2_tuple is not None:
                    k2 = max(k2_tuple[0], int(n*k2_tuple[1]))
                else:
                    k2 = None
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio_continuous, portfolio_evalute_wSol, rng_alg, *prob_args, k2 = k2)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, gurobi_portfolio_continuous, portfolio_evalute_wSol, rng_alg, *prob_args, k2 = k2)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, k2 = {k2}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
        
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
    


def comparison_test(k_list, B1_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    SAA_list = []
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B1_list))]
    bagging_alg3_trick_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B1_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B1_list))]
        bagging_alg3_trick_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B1_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_portfolio(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote_LP(sample_n, 1, n, gurobi_portfolio_continuous, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([float(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B1 in enumerate(B1_list):
                    tic = time.time()
                    bagging_alg3, _, bagging_alg3_trick, _ = test_baggingTwoPhase_woSplit_LP(sample_n, B1, k, gurobi_portfolio_continuous, portfolio_evaluate_exact, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3]))
                    bagging_alg3_trick_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3_trick]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}")

        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B1_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg3_trick_list[ind1][ind2].append(bagging_alg3_trick_intermediate[ind1][ind2])
        
    return SAA_list, bagging_alg3_list, bagging_alg3_trick_list

def evaluation_test(SAA_list, bagging_alg3_list, bagging_alg3_trick_list, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    k_list_len = len(bagging_alg3_list[0])
    B1_list_len = len(bagging_alg3_list)

    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B1_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B1_list_len)]
    bagging_alg3_trick_obj_list, bagging_alg3_trick_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B1_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B1_list_len)]

    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = portfolio_evaluate_exact(SAA_list[i][j], *prob_args)
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B1_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_3_trick = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = portfolio_evaluate_exact(bagging_alg3_list[ind1][ind2][i][j], *prob_args)
                    bagging_obj_3_trick = portfolio_evaluate_exact(bagging_alg3_trick_list[ind1][ind2][i][j], *prob_args)
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_3_trick.append(bagging_obj_3_trick)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg3_trick_obj_list[ind1][ind2].append(current_bagging_obj_list_3_trick)
                bagging_alg3_trick_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3_trick))
    
    return SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg3_trick_obj_list, bagging_alg3_trick_obj_avg



######################## eta comparison ########################
def solution_retrieval(B, k, n, sample_args, mu, b):
    # need to ensure the retrieved set is large enough by taking a large B
    rng = np.random.default_rng(seed = 1)
    retrieved_set = set()
    sample_n = genSample_portfolio(n, rng, type = sample_args['type'], params = sample_args['params'])
    _, retrieved_set = majority_vote_LP(sample_n, B, k, gurobi_portfolio_continuous, rng, mu, b)
    retrieved_set = set([tuple(float(x) for x in sol) for sol in retrieved_set])
    return retrieved_set

def solution_evaluation(retrieved_set, mu, b):
    # input a set of solutions, output a dictionary of true objective values
    obj_values = {}
    for x in retrieved_set:
        obj_values[x] = portfolio_evaluate_exact(x, mu, b)
    return obj_values

def solution_SAA_evaluation(retrieved_set, k, number_of_iterations, rng_sample, sample_args, mu, b):
    # treat the problem in the minimization form
    # input a set of solutions, assuming they are the entire decision space
    # use SAA to evaluate the solutions, and return a dictionary of list of objective values
    # the key is the iteration number
    x_count_dict = {iter: {x: None for x in retrieved_set} for iter in range(number_of_iterations)}
    obj_opt_dict = {iter: None for iter in range(number_of_iterations)}
    SAA_conditional_pk = {x: 0 for x in retrieved_set}
    for iter in range(number_of_iterations):
        sample_n = genSample_portfolio(k, rng_sample, type = sample_args['type'], params = sample_args['params'])
        obj_opt = np.inf
        x_min = None
        for x in retrieved_set:
            obj_x = - portfolio_evalute_wSol(sample_n, x, mu, b)[0]
            x_count_dict[iter][x] = obj_x
            if obj_x < obj_opt:
                obj_opt = obj_x
                x_min = x
        obj_opt_dict[iter] = obj_opt
        SAA_conditional_pk[x_min] += 1/number_of_iterations
    return x_count_dict, obj_opt_dict, SAA_conditional_pk


def simulate_Alg34_pk(retrieved_set, x_count_dict, obj_opt_dict, epsilon):
    # compute p_k(x) for Alg3 and Alg4, which is the probability of including x in the suboptimal set
    # x_count_dict it the SAA simulated objective value for each x
    # obj_opt_dict is the SAA simulated optimal objective value
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
        if x_count[x] < obj_opt + epsilon:
            suboptimal_set.add(x)
    return suboptimal_set

def get_suboptimal_set(obj_values, obj_opt, delta):
    # get delta-suboptimal solutions based on obj_values dictionary
    # obj_opt = min(obj_values.values()) # assume the optimal solution is in the set
    # essentially this is the same function as get_SAA_suboptimal_set
    suboptimal_set = set()
    for x in obj_values:
        if obj_values[x] < obj_opt + delta:
            suboptimal_set.add(x)
    return suboptimal_set


def portfolio_eta_comparison(delta_list, B, k, epsilon, n, number_of_iterations, rng_sample, sample_args, mu, b):
    # n: sample_number for solution retrieval (large)
    # k: sample number used for eta simulation (small)
    # B: control the number of solutions retrieved
    # we need number of iterations to be significantly greater than B
    # for Alg1, just simulates p_k(x), and compute max p_k(x) - max_{x not in delta optimal set} p_k(x)
    # for Alg3 and Alg4, this p_k(x) is the probability to be included in the SAA suboptimal set
    retrieved_set = solution_retrieval(B, k, n, sample_args, mu, b)
    x_count_dict, obj_opt_dict, SAA_conditional_pk = solution_SAA_evaluation(retrieved_set, k, number_of_iterations, rng_sample, sample_args, mu, b)
    pk_dict_Alg34 = simulate_Alg34_pk(retrieved_set, x_count_dict, obj_opt_dict, epsilon)
    max_pk_SAA = max(SAA_conditional_pk.values())
    max_pk_Alg34 = max(pk_dict_Alg34.values())

    obj_values = solution_evaluation(retrieved_set, mu, b) # compute the true objective values
    obj_opt = min(obj_values.values())

    SAA_eta_list = []
    Alg34_eta_list = []
    for delta in delta_list:
        suboptimal_set_delta = get_suboptimal_set(obj_values, obj_opt, delta)
        not_suboptimal_set_delta = retrieved_set - suboptimal_set_delta
        max_pk_SAA_delta = 0
        for x in not_suboptimal_set_delta:
            max_pk_SAA_delta = max(max_pk_SAA_delta, SAA_conditional_pk[x])
        
        max_pk_Alg34_delta = 0
        for x in not_suboptimal_set_delta:
            max_pk_Alg34_delta = max(max_pk_Alg34_delta, pk_dict_Alg34[x])
        
        SAA_eta_list.append(max_pk_SAA - max_pk_SAA_delta)
        Alg34_eta_list.append(max_pk_Alg34 - max_pk_Alg34_delta)
    
    return SAA_eta_list, Alg34_eta_list, retrieved_set, x_count_dict, obj_opt_dict, SAA_conditional_pk, pk_dict_Alg34
