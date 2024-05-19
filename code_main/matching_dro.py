import time
import json
import numpy as np
from utils.matching_functions import matching_obj_optimal, generateW, comparison_DRO, evaluation_DRO
from utils.plotting import plot_droComparison, plot_CI_droComparison, plot_optGap_droComparison

if __name__ == "__main__":
    seed = 2024
    N = 5
    
    w = {(0, 0): 1.8865364093012886,
 (0, 1): 2.398555301120877,
 (0, 2): 1.8306427790927857,
 (0, 3): 1.7492991192565832,
 (0, 4): 2.3237863958023786,
 (1, 0): 2.137122473241966,
 (1, 1): 2.2498292819020653,
 (1, 2): 1.8709355265561154,
 (1, 3): 1.7336844551004142,
 (1, 4): 1.8512613494646823,
 (2, 0): 1.873453484656252,
 (2, 1): 2.32957861213437,
 (2, 2): 2.2815754847013983,
 (2, 3): 2.1955418952557166,
 (2, 4): 1.9664292773529026,
 (3, 0): 1.6368204540890734,
 (3, 1): 2.1733533180049087,
 (3, 2): 2.29142702055407,
 (3, 3): 1.64693564175383,
 (3, 4): 2.2760005110017376,
 (4, 0): 2.390491551306702,
 (4, 1): 2.340076629674212,
 (4, 2): 1.8125406416083787,
 (4, 3): 1.9427529395181724,
 (4, 4): 1.6101934594984615}
    

    sample_args = {
        "type" : "pareto",
        # "type" : "sym_pareto",
        # "params": [2,2,2,2,2,2]
        # "params": [1.9,1.9,1.9,1.9,1.9,1.9]
         "params": [
   2.127985257580268,
   1.7150235669612395,
   2.203886947758919,
   1.8794127488738956,
   1.8001092745067868,
   2.148190617455044,
   2.1448806577187685,
   1.75211799503506,
   1.7802895825781155
  ]
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200]
    k_list = [(10, 0.005)]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 0.001
    varepsilon_list = [2**i for i in range(-6,2)] 
    # varepsilon_list =  [10, 20, 50, 100, 1000]
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(10, 16)])

    # testing parameters
    # B_list = [2,3]
    # k_list = [10]
    # B12_list = [(2,3),(3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.001
    # varepsilon_list = [2**i for i in range(-6,-4)]
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    tic = time.time()
    SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_DRO(B_list, k_list, B12_list, epsilon, tolerance, varepsilon_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, N, w.copy())
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "dro_wasserstein_list": dro_wasserstein_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    SAA_obj_list, SAA_obj_avg, dro_wasserstein_obj_list, dro_wasserstein_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w.copy())
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "dro_wasserstein_obj_list": dro_wasserstein_obj_list, "dro_wasserstein_obj_avg": dro_wasserstein_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    plot_droComparison(SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)
    plot_CI_droComparison(SAA_obj_list, dro_wasserstein_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

    obj_opt, _ = matching_obj_optimal(sample_args, N, w)
    # print(obj_opt)
    plot_optGap_droComparison(obj_opt, "max", SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

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
        "varepsilon_list": varepsilon_list,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f)

