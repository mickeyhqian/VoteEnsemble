import numpy as np
import time
import json
from utils.portfolio_functions import portfolio_prob_comparison
from utils.plotting import plot_probComparison

if __name__ == "__main__":
    seed = 2024
    m = 6

    mu = np.array(
        [
    4.075363139480371,
    3.0930271431089653,
    2.7280699955747396,
    3.040554341010973,
    1.5851363596432098,
    2.761708917626271
    # 4.427214001663112,
    # 2.2444313895434127,
    # 3.295879016752802,
    # 2.324773020139842,
    # 2.41364867026627,
    # 3.1157331535368233
  ]
    )

    p = np.array(
        [
    0.9700751597162569,
    0.9771104636723029,
    1.0557778081173659,
    0.8934521756950546,
    1.280249457185443,
    0.39030737697894446
    # 0.13771753159888833,
    # 1.8345420923650477,
    # 0.1718677444709127,
    # 1.128175938730182,
    # 1.0612742642800843,
    # 0.31549275985232983
  ])
    
    b = 1.8403077686276639
    # b = 1.4347461101403558

    sample_args = {
        "type" : "sym_pareto",
        "params": mu.tolist()
    }

    alpha = 0.95

    B_list = [200]
    k_list = [0.1, 10]
    B12_list = [(20,200)]
    epsilon = 'dynamic'
    tolerance = 0.005
    number_of_iterations = 200
    sample_number = np.array([2**i for i in range(5, 12)])
    large_number_sample = 1000000

    # test
    # B_list = [2,3]
    # k_list = [0.1, 2, 50]
    # B12_list = [(2,4), (3,6)]
    # epsilon = 'dynamic'
    # tolerance = 0.005
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])
    # large_number_sample = 10000

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    tic = time.time()
    SAA_prob_opt_list, SAA_prob_dist_list, baggingAlg1_prob_opt_list, baggingAlg1_prob_dist_list, baggingAlg3_prob_opt_list, baggingAlg3_prob_dist_list, baggingAlg4_prob_opt_list, baggingAlg4_prob_dist_list = portfolio_prob_comparison(B_list, k_list, B12_list, epsilon, tolerance, sample_number, large_number_sample, number_of_iterations, rng_sample, rng_alg, sample_args, p, mu, b, alpha)
    print(f"Total time: {time.time()-tic}")

    with open("portfolio_prob_comparison.json", 'w') as f:
        json.dump({"SAA_prob_opt_list": SAA_prob_opt_list, "SAA_prob_dist_list": SAA_prob_dist_list, "baggingAlg1_prob_opt_list": baggingAlg1_prob_opt_list, "baggingAlg1_prob_dist_list": baggingAlg1_prob_dist_list, "baggingAlg3_prob_opt_list": baggingAlg3_prob_opt_list, "baggingAlg3_prob_dist_list": baggingAlg3_prob_dist_list, "baggingAlg4_prob_opt_list": baggingAlg4_prob_opt_list, "baggingAlg4_prob_dist_list": baggingAlg4_prob_dist_list}, f)

    for ind2, k in enumerate(k_list):
        for ind1, B in enumerate(B_list):
            plot_probComparison(SAA_prob_opt_list, baggingAlg1_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B}_{k}_Alg1')
        
        for ind1, (B1,B2) in enumerate(B12_list):
            plot_probComparison(SAA_prob_opt_list, baggingAlg3_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B1}_{B2}_{k}_Alg3')
            plot_probComparison(SAA_prob_opt_list, baggingAlg4_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B1}_{B2}_{k}_Alg4')
        
    parameters = {
        "seed": seed,
        "m": m,
        "mu": mu.tolist(),
        "p": p.tolist(),
        "b": b,
        "alpha": alpha,
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "large_number_sample": large_number_sample
    }

    with open("portfolio_parameters.json", 'w') as f:
        json.dump(parameters, f, indent=2)
