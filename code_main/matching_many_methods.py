import time
import json
import numpy as np
from utils.matching_functions import comparison_many_methods, evaluation_many_methods, matching_obj_optimal
from utils.plotting import plot_many_methods, plot_CI_many_methods

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

    B = 200
    k_tuple = (10, 0.005)
    B12 = (20,200)
    epsilon = "dynamic"
    tolerance = 0.001
    varepsilon = 2**-2
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(10, 16)])

    # tic = time.time()
    # SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list = comparison_many_methods(B,k_tuple,B12,epsilon,tolerance,varepsilon,number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, N, w.copy())
    # with open("solution_lists.json", "w") as f:
    #     json.dump({"SAA_list": SAA_list, "dro_wasserstein_list": dro_wasserstein_list, "bagging_alg1_SAA_list": bagging_alg1_SAA_list, "bagging_alg1_DRO_list": bagging_alg1_DRO_list, "bagging_alg3_SAA_list": bagging_alg3_SAA_list, "bagging_alg3_DRO_list": bagging_alg3_DRO_list, "bagging_alg4_SAA_list": bagging_alg4_SAA_list, "bagging_alg4_DRO_list": bagging_alg4_DRO_list}, f)

    # evaluation_results = evaluation_many_methods(SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list, sample_args, N, w.copy())
    # with open("obj_lists.json", "w") as f:
    #     json.dump(evaluation_results, f)


    # print(f"Total time: {time.time()-tic}")

    # avg_obj = {
    #     "SAA": evaluation_results["SAA_obj_avg"],
    #     "DRO": evaluation_results["dro_wasserstein_obj_avg"],
    #     "Alg1+SAA": evaluation_results["bagging_alg1_SAA_obj_avg"],
    #     "Alg1+DRO": evaluation_results["bagging_alg1_DRO_obj_avg"],
    #     "Alg3+SAA": evaluation_results["bagging_alg3_SAA_obj_avg"],
    #     "Alg3+DRO": evaluation_results["bagging_alg3_DRO_obj_avg"],
    #     "Alg4+SAA": evaluation_results["bagging_alg4_SAA_obj_avg"],
    #     "Alg4+DRO": evaluation_results["bagging_alg4_DRO_obj_avg"]
    # }

    # plot_many_methods(avg_obj, np.log2(sample_number))

    # list_obj = {
    #     "SAA": evaluation_results["SAA_obj_list"],
    #     "DRO": evaluation_results["dro_wasserstein_obj_list"],
    #     "Alg1+SAA": evaluation_results["bagging_alg1_SAA_obj_list"],
    #     "Alg1+DRO": evaluation_results["bagging_alg1_DRO_obj_list"],
    #     "Alg3+SAA": evaluation_results["bagging_alg3_SAA_obj_list"],
    #     "Alg3+DRO": evaluation_results["bagging_alg3_DRO_obj_list"],
    #     "Alg4+SAA": evaluation_results["bagging_alg4_SAA_obj_list"],
    #     "Alg4+DRO": evaluation_results["bagging_alg4_DRO_obj_list"]
    # }
    # plot_CI_many_methods(list_obj, np.log2(sample_number))

    obj_opt, _ = matching_obj_optimal(sample_args, N, w)

    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
        "sample_args": sample_args,
        "B": B,
        "k_tuple": k_tuple,
        "B12": B12,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "varepsilon": varepsilon,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f)