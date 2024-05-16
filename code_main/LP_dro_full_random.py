import time
import json
import numpy as np
from utils.LP_functions_full_random import *
from utils.plotting import plot_droComparison, plot_CI_droComparison, plot_optGap_droComparison


if __name__ == "__main__":
    seed = 2024
    N = 8

    w = {(1, 2): 3,
    (1, 3): 3,
    (1, 4): 3,
    (1, 5): 3.5,
    (1, 6): 3.5,
    (1, 7): 3.5,
    (1, 8): 3.5,
    (2, 3): 3,
    (2, 4): 3,
    (2, 5): 3.5,
    (2, 6): 3.5,
    (2, 7): 3.5,
    (2, 8): 3.5,
    (3, 4): 3,
    (3, 5): 3.5,
    (3, 6): 3.5,
    (3, 7): 3.5,
    (3, 8): 3.5,
    (4, 5): 3.5,
    (4, 6): 3.5,
    (4, 7): 3.5,
    (4, 8): 3.5,
    (5, 6): 3.5,
    (5, 7): 3.5,
    (5, 8): 3.5,
    (6, 7): 3.5,
    (6, 8): 3.5,
    (7, 8): 3.5}

    params = get_pareto_params(N,w)
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])

    sample_args = {
        "type" : "pareto",
        "params": params
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200]
    k_list = [(10, 0.005)]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 0.001
    varepsilon_list = [2**i for i in range(-6,4)]
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(10, 16)])

    tic = time.time()
    SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_DRO(B_list, k_list, B12_list, epsilon, tolerance, varepsilon_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, N, w, A)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "dro_wasserstein_list": dro_wasserstein_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)

    # with open("solution_lists.json", "r") as f:
    #     data = json.load(f)
    #     SAA_list = data["SAA_list"]
    #     dro_wasserstein_list = data["dro_wasserstein_list"]
    #     bagging_alg1_list = data["bagging_alg1_list"]
    #     bagging_alg3_list = data["bagging_alg3_list"]
    #     bagging_alg4_list = data["bagging_alg4_list"]
    
    # # convert lists to tuples
    # for i in range(len(sample_number)):
    #     for j in range(number_of_iterations):
    #         SAA_list[i][j] = tuple(SAA_list[i][j])
    #         for ind in range(len(varepsilon_list)):
    #             dro_wasserstein_list[ind][i][j] = tuple(dro_wasserstein_list[ind][i][j])
    #         for ind1 in range(len(B_list)):
    #             for ind2 in range(len(k_list)):
    #                 bagging_alg1_list[ind1][ind2][i][j] = tuple(bagging_alg1_list[ind1][ind2][i][j])
    #         for ind1 in range(len(B12_list)):
    #             for ind2 in range(len(k_list)):
    #                 bagging_alg3_list[ind1][ind2][i][j] = tuple(bagging_alg3_list[ind1][ind2][i][j])
    #                 bagging_alg4_list[ind1][ind2][i][j] = tuple(bagging_alg4_list[ind1][ind2][i][j])

    SAA_obj_list, SAA_obj_avg, dro_wasserstein_obj_list, dro_wasserstein_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w, A)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "dro_wasserstein_obj_list": dro_wasserstein_obj_list, "dro_wasserstein_obj_avg": dro_wasserstein_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)

    plot_droComparison(SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)
    plot_CI_droComparison(SAA_obj_list, dro_wasserstein_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

    obj_opt, _ = LP_obj_optimal(N, w, A)
    plot_optGap_droComparison(obj_opt, "max", SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
        "A": A.tolist(),
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "varepsilon_list": varepsilon_list,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)