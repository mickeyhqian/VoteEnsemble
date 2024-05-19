from ParallelSolve import gurobi_SSKP, majority_vote, prob_simulate_SSKP
import time
import numpy as np
from utils.generateSamples import genSample_SSKP
import json
from utils.plotting import plot_probComparison
from utils.SSKP_functions import SSKP_prob_comparison


def SSKP_prob(sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args):
    # function that calculates the probability \hat p(x) for all x.
    # input: sample_number, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args
    # output: \hat p(x*), \hat p(x*) - max_{x\neq x*} \hat p(x), distribution of \hat p(x)

    ########## parts that need to be modified for different problems ##########
    sample_large = genSample_SSKP(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
    SAA, _ = majority_vote(sample_large, 1, large_number_sample, gurobi_SSKP, rng_alg, *prob_args)
    ##########
    
    x_star = SAA if type(SAA) == int else tuple(int(entry) for entry in SAA)

    prob_opt_list = []
    prob_diff_list = []
    prob_dist_list = []
    for n in sample_number:
        # use a single function to complete the following loops in parallel
        count = prob_simulate_SSKP(n, number_of_iterations, rng_sample, sample_args, *prob_args)
        
        prob_opt_list.append(count.get(x_star, 0))
        max_prob = 0
        for key in count:
            if key != x_star:
                max_prob = max(max_prob, count[key])
        prob_diff_list.append(count[x_star] - max_prob)
        prob_dist_list.append(count)
    
    return prob_opt_list, prob_diff_list, prob_dist_list, x_star


if __name__ == "__main__":
    # r = [3.2701236422941093, 3.3207149493214994, 3.556858029428708]
    # c, q = 3.7856629820554946, 1.7096129150007453
    # sample_args = {
    #         'type': 'pareto',
    #         'params': [2.0033248484659976, 1.9462659915572313, 2.0148555044660448]
    #     }

    r = [
        3.2701236422941093,
        3.3207149493214994,
        3.556858029428708
    ]
    c = 3.7856629820554946
    q = 1.7096129150007453
    sample_args = {
        "type": "pareto",
        "params": [
        1.966,
        2.21,
        1.89
        ]
    }

    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)    

    B_list = [1]
    k_list = [(1,0)]
    number_of_iterations = 200
    # sample_number = np.array([2**i for i in range(5, 12)])
    sample_number = np.array([2**i for i in range(1,7)])
    large_number_sample = 200000

    tic = time.time()
    SAA_prob_opt_list, SAA_prob_diff_list, SAA_prob_dist_list, bagging_prob_opt_list, bagging_prob_diff_list, bagging_prob_dist_list, x_star = SSKP_prob_comparison(B_list, k_list, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, r, c, q)
    print(f"Total time: {time.time()-tic}")
    
    with open("SSKP_prob_comparison.json", "w") as f:
        json.dump({"SAA_prob_opt_list": SAA_prob_opt_list, "SAA_prob_diff_list": SAA_prob_diff_list, "SAA_prob_dist_list": SAA_prob_dist_list, "bagging_prob_opt_list": bagging_prob_opt_list, "bagging_prob_diff_list": bagging_prob_diff_list, "bagging_prob_dist_list": bagging_prob_dist_list, "x_star": x_star}, f)

    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            plot_probComparison(SAA_prob_opt_list, bagging_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B}_{k}')
            # plot_probComparison(SAA_prob_diff_list, bagging_prob_diff_list[ind1][ind2], sample_number, f'prob_diff_{B}_{k}')
            plot_probComparison(-np.log([1-x for x in SAA_prob_opt_list]), -np.log([1-x for x in bagging_prob_opt_list[ind1][ind2]]), sample_number, f'log_prob_opt_{B}_{k}')

    parameters = {
        "seed": seed,
        "r": r,
        "c": c,
        "q": q,
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "large_number_sample": large_number_sample
    }
    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)





