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


def generate_instances_new(N, w, instance_num, gap, A):
    prob_args_list = []
    sample_args_list = []
    for i in range(instance_num):
        w_new = {}
        for key in w:
            w_new[key] = w[key] + gap*i
        params = get_pareto_params(N,w_new)
        sample_args = {
            "type" : "pareto",
            "params": params
        }
        prob_args_list.append((N, w_new, A))
        sample_args_list.append(sample_args)
    return prob_args_list, sample_args_list
        
    
if __name__ == "__main__":
    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    N = 8
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    
    # multi-optimal instance
    # base_list = [3, 3.5, 4, 4.5, 5, 5.5]
    # base_list = [4]
    # gap = 0.5
    # prob_args_list, sample_args_list = generate_instances(N, base_list, gap, A)

    w = {(1, 2): 1.1,
 (1, 3): 1.2,
 (1, 4): 1.3,
 (1, 5): 1.4,
 (1, 6): 1.5,
 (1, 7): 1.6,
 (1, 8): 1.7,
 (2, 3): 1.8,
 (2, 4): 1.9,
 (2, 5): 2.0,
 (2, 6): 2.1,
 (2, 7): 2.2,
 (2, 8): 2.3,
 (3, 4): 2.4,
 (3, 5): 2.5,
 (3, 6): 2.6,
 (3, 7): 2.7,
 (3, 8): 2.8,
 (4, 5): 2.9,
 (4, 6): 3.0,
 (4, 7): 3.1,
 (4, 8): 3.2,
 (5, 6): 3.3,
 (5, 7): 3.4,
 (5, 8): 3.5,
 (6, 7): 3.6,
 (6, 8): 3.7,
 (7, 8): 3.8}
    
    gap = 0.5
    instance_num = 6
    prob_args_list, sample_args_list = generate_instances_new(N, w, instance_num, gap, A)

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
    
    # parameters = {
    #     "seed": seed,
    #     "N": N,
    #     "A": A.tolist(),
    #     "base_list": base_list,
    #     "gap": gap,
    #     "B": B,
    #     "k_tuple": k_tuple,
    #     "B12": B12,
    #     "epsilon": epsilon,
    #     "tolerance": tolerance,
    #     "number_of_iterations": number_of_iterations,
    #     "n": n
    # }

    parameters = {
        "seed": seed,
        "N": N,
        "A": A.tolist(),
        "w": {str(key): value for key, value in w.items()},
        "gap": gap,
        "instance_num": instance_num,
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

