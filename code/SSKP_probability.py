from ParallelSolve import gurobi_SSKP, majority_vote, prob_simulate_SSKP
import time
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from utils.SSKP_functions import genSample_SSKP
import json


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


def SSKP_prob_comparison(B_list, k_list, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, *prob_args):
    # function that compare the probability of outputting the optimal solution for SAA and Bagging-SAA
    sample_large = genSample_SSKP(large_number_sample, rng_sample, type = sample_args['type'], params = sample_args['params'])
    SAA, _ = majority_vote(sample_large, 1, large_number_sample, gurobi_SSKP, rng_alg, *prob_args)

    x_star = str(SAA) if type(SAA) == int else str(tuple(int(entry) for entry in SAA))
    print("Optimal solution: ", x_star)

    SAA_prob_opt_list = []
    SAA_prob_diff_list = []
    SAA_prob_dist_list = []
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

def plot_comparison(dist_1, dist_2, sample_number, name):
    n_groups = len(sample_number)
    bar_width = 0.35  # Width of the bars
    index = np.arange(n_groups)  # Position of groups

    fig, ax = plt.subplots()

    bars1 = ax.bar(index - bar_width/2, dist_1, bar_width, label='Setting 1', color='b', edgecolor='black')
    bars2 = ax.bar(index + bar_width/2, dist_2, bar_width, label='Setting 2', color='r', edgecolor='black')

    ax.set_xlabel('Sample Number')
    if name == 'prob_opt':
        ax.set_ylabel('Probability')
    elif name == 'prob_diff':
        ax.set_ylabel('Probability Difference')
    ax.set_title(name)
    ax.set_xticks(index)
    ax.set_xticklabels(sample_number)
    ax.legend()
    # plt.show()
    plt.savefig(name + '.png')

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

    B_list = [200]
    k_list = [0.1, 2]
    number_of_iterations = 200
    sample_number = np.array([2**i for i in range(5, 12)])
    large_number_sample = 200000

    SAA_prob_opt_list, SAA_prob_diff_list, SAA_prob_dist_list, bagging_prob_opt_list, bagging_prob_diff_list, bagging_prob_dist_list, x_star = SSKP_prob_comparison(B_list, k_list, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, r, c, q)

    with open("SSKP_prob_comparison.json", "w") as f:
        json.dump({"SAA_prob_opt_list": SAA_prob_opt_list, "SAA_prob_diff_list": SAA_prob_diff_list, "SAA_prob_dist_list": SAA_prob_dist_list, "bagging_prob_opt_list": bagging_prob_opt_list, "bagging_prob_diff_list": bagging_prob_diff_list, "bagging_prob_dist_list": bagging_prob_dist_list, "x_star": x_star}, f)

    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            plot_comparison(SAA_prob_opt_list, bagging_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B}_{k}')
            # plot_comparison(SAA_prob_diff_list, bagging_prob_diff_list[ind1][ind2], sample_number, f'prob_diff_{B}_{k}')
            plot_comparison(-np.log([1-x for x in SAA_prob_opt_list]), -np.log([1-x for x in bagging_prob_opt_list[ind1][ind2]]), sample_number, f'log_prob_opt_{B}_{k}')

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





