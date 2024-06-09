import numpy as np
import json
import time
from utils.LP_functions_full_random import check_SAA_optimal, simulate_SAA_pk, get_pareto_params

# utility script. used to check if appropriate sizes of resample are enough

if __name__ == "__main__":
    N = 8
    w = {(1, 2): 2.5,
    (1, 3): 2.5,
    (1, 4): 2.5,
    (1, 5): 3,
    (1, 6): 3,
    (1, 7): 3,
    (1, 8): 3,
    (2, 3): 2.5,
    (2, 4): 2.5,
    (2, 5): 3,
    (2, 6): 3,
    (2, 7): 3,
    (2, 8): 3,
    (3, 4): 2.5,
    (3, 5): 3,
    (3, 6): 3,
    (3, 7): 3,
    (3, 8): 3,
    (4, 5): 3,
    (4, 6): 3,
    (4, 7): 3,
    (4, 8): 3,
    (5, 6): 3,
    (5, 7): 3,
    (5, 8): 3,
    (6, 7): 3,
    (6, 8): 3,
    (7, 8): 3}

    params = get_pareto_params(N,w)

    sample_args = {
        "type" : "pareto",
        "params": params
    }

    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])

    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    
    number_of_iterations = 500
    result = []

    for n in [10, 50, 100]:
        pk_dict, retrieved_set = simulate_SAA_pk(n, number_of_iterations, rng_sample, sample_args, N, w, A)
        prob_opt = check_SAA_optimal(pk_dict, retrieved_set, N, w ,A)
        result.append(prob_opt)
    
    print(result)
    


