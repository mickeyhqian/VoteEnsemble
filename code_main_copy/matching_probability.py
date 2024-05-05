import numpy as np
import time
import json
from utils.matching_functions import matching_prob_comparison
from utils.plotting import plot_probComparison


if __name__ == "__main__":
    seed = 2024
    N = 8
    w = {(1, 2): None,
        (1, 3): None,
        (1, 4): None,
        (1, 5): 1.9119257287734244,
        (1, 6): 1.6228257372263202,
        (1, 7): 1.7968885916853101,
        (1, 8): 2.313248853705918,
        (2, 3): None,
        (2, 4): None,
        (2, 5): 1.9362072829402424,
        (2, 6): 2.451003072707256,
        (2, 7): 2.398840628680899,
        (2, 8): 1.8209819889450793,
        (3, 4): None,
        (3, 5): 1.9978624908336502,
        (3, 6): 2.0670783259937506,
        (3, 7): 1.8577180932652504,
        (3, 8): 2.3907208610757693,
        (4, 5): 2.203256964324676,
        (4, 6): 2.422592793261833,
        (4, 7): 1.6985434453495578,
        (4, 8): 1.9787435233099542,
        (5, 6): 2,
        (5, 7): 2,
        (5, 8): 2,
        (6, 7): 2,
        (6, 8): 2,
        (7, 8): 2}
    
    sample_args = {
        "type" : "pareto",
        "params": [2,2,2,2,2,2]
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200, 400]
    k_list = [0.1, 2, 10]
    B12_list = [(20,200), (20,400)]
    epsilon = "dynamic"
    tolerance = 0.005
    number_of_iterations = 200
    sample_number = np.array([2**i for i in range(5, 12)])
    
    # test parameters
    # B_list = [2,3]
    # k_list = [0.1, 2]
    # B12_list = [(2,3), (3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.005
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    tic = time.time()
    SAA_prob_opt_list, SAA_prob_dist_list, baggingAlg1_prob_opt_list, baggingAlg1_prob_dist_list, baggingAlg3_prob_opt_list, baggingAlg3_prob_dist_list, baggingAlg4_prob_opt_list, baggingAlg4_prob_dist_list = matching_prob_comparison(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, N, w)
    print(f"Total time: {time.time()-tic}")

    with open("matching_prob_comparison.json", "w") as f:
        json.dump({"SAA_prob_opt_list": SAA_prob_opt_list, "SAA_prob_dist_list": SAA_prob_dist_list, "baggingAlg1_prob_opt_list": baggingAlg1_prob_opt_list, "baggingAlg1_prob_dist_list": baggingAlg1_prob_dist_list, "baggingAlg3_prob_opt_list": baggingAlg3_prob_opt_list, "baggingAlg3_prob_dist_list": baggingAlg3_prob_dist_list, "baggingAlg4_prob_opt_list": baggingAlg4_prob_opt_list, "baggingAlg4_prob_dist_list": baggingAlg4_prob_dist_list}, f)
    
    for ind2, k in enumerate(k_list):
        for ind1, B in enumerate(B_list):
            plot_probComparison(SAA_prob_opt_list, baggingAlg1_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B}_{k}_Alg1')
        
        for ind1, (B1,B2) in enumerate(B12_list):
            plot_probComparison(SAA_prob_opt_list, baggingAlg3_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B1}_{B2}_{k}_Alg3')
            plot_probComparison(SAA_prob_opt_list, baggingAlg4_prob_opt_list[ind1][ind2], sample_number, f'prob_opt_{B1}_{B2}_{k}_Alg4')
    
    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist()
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)