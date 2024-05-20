import time
import json
import numpy as np
from utils.LP_functions_full_random import *

def generateW(N, base, gap):
    w = {}
    for i in range(1, N):
        for j in range(i+1, N+1):
            if i <= 3 and j <= 4:
                w[(i,j)] = base
            else:
                w[(i,j)] = base + gap
    return w
            

def generate_instances(N, base_list, gap, A):
    prob_args_list = []
    sample_args_list = []
    for base in base_list:
        w = generateW(N, base, gap)
        params = get_pareto_params(N,w)
        sample_args = {
            "type" : "pareto",
            "params": params
        }
        prob_args_list.append((N, w, A))
        sample_args_list.append(sample_args)
    return prob_args_list, sample_args_list

    

if __name__ == "__main__":
    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    N = 8
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    base_list = [3, 3.5, 4, 4.5, 5, 5.5]
    # base_list = [4]
    gap = 0.5
    prob_args_list, sample_args_list = generate_instances(N, base_list, gap, A)

    # test
    B = [2000]
    k_tuple = (50, 0)
    B12 = [(200,5000)]
    epsilon = 'dynamic'
    tolerance = 0.01
    number_of_iterations = 200
    n = 1000000

    SAA_prob_list, bagging_alg1_prob_list, bagging_alg3_prob_list, bagging_alg4_prob_list = tail_influence(B, k_tuple, B12, epsilon, tolerance, number_of_iterations, n, rng_sample, rng_alg, sample_args_list, prob_args_list)

    with open("tail_influence_solution_lists.json", "w") as f:
        json.dump({"SAA_prob_list": SAA_prob_list, "bagging_alg1_prob_list": bagging_alg1_prob_list, "bagging_alg3_prob_list": bagging_alg3_prob_list, "bagging_alg4_prob_list": bagging_alg4_prob_list}, f)
    
    parameters = {
        "seed": seed,
        "N": N,
        "A": A.tolist(),
        "base_list": base_list,
        "gap": gap,
        "B": B,
        "k_tuple": k_tuple,
        "B12": B12,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "n": n
    }

    with open("tail_influence_parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)


    

    

