from utils.SSKP_functions import comparison_many_methods, evaluation_many_methods
from utils.plotting import plot_many_methods, plot_CI_many_methods
import time
import numpy as np
import json

if __name__ == "__main__":
    r = [3.2701236422941093, 3.3207149493214994, 3.556858029428708]
    c, q = 3.7856629820554946, 1.7096129150007453
    sample_args = {
            'type': 'pareto',
            'params': [2.0033248484659976, 1.9462659915572313, 2.0148555044660448]
        }
    
    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B = 200
    k = 10
    B12 = (20,200)
    epsilon = "dynamic"
    tolerance = 0.005
    varepsilon = 0.1
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(7, 11)])
    large_number_sample = 500000
    eval_time = 10

    # testing parameters
    # B = 2
    # k = 10
    # B12 = (2,3)
    # epsilon = "dynamic"
    # tolerance = 0.005
    # varepsilon = 0.1
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])
    # large_number_sample = 1000
    # eval_time = 2

    tic = time.time()
    SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list = comparison_many_methods(B,k,B12,epsilon,tolerance,varepsilon,number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, r, c, q)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "dro_wasserstein_list": dro_wasserstein_list, "bagging_alg1_SAA_list": bagging_alg1_SAA_list, "bagging_alg1_DRO_list": bagging_alg1_DRO_list, "bagging_alg3_SAA_list": bagging_alg3_SAA_list, "bagging_alg3_DRO_list": bagging_alg3_DRO_list, "bagging_alg4_SAA_list": bagging_alg4_SAA_list, "bagging_alg4_DRO_list": bagging_alg4_DRO_list}, f)
    
    evaluation_results = evaluation_many_methods(SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list, large_number_sample, eval_time, rng_sample, sample_args, r, c, q)
    with open("obj_lists.json", "w") as f:
        json.dump(evaluation_results, f)
    
    print(f"Total time: {time.time()-tic}")

    avg_obj = {
        "SAA": evaluation_results["SAA_obj_avg"],
        "DRO": evaluation_results["dro_wasserstein_obj_avg"],
        "Alg1+SAA": evaluation_results["bagging_alg1_SAA_obj_avg"],
        "Alg1+DRO": evaluation_results["bagging_alg1_DRO_obj_avg"],
        "Alg3+SAA": evaluation_results["bagging_alg3_SAA_obj_avg"],
        "Alg3+DRO": evaluation_results["bagging_alg3_DRO_obj_avg"],
        "Alg4+SAA": evaluation_results["bagging_alg4_SAA_obj_avg"],
        "Alg4+DRO": evaluation_results["bagging_alg4_DRO_obj_avg"]
    }

    plot_many_methods(avg_obj, np.log2(sample_number))

    list_obj = {
        "SAA": evaluation_results["SAA_obj_list"],
        "DRO": evaluation_results["dro_wasserstein_obj_list"],
        "Alg1+SAA": evaluation_results["bagging_alg1_SAA_obj_list"],
        "Alg1+DRO": evaluation_results["bagging_alg1_DRO_obj_list"],
        "Alg3+SAA": evaluation_results["bagging_alg3_SAA_obj_list"],
        "Alg3+DRO": evaluation_results["bagging_alg3_DRO_obj_list"],
        "Alg4+SAA": evaluation_results["bagging_alg4_SAA_obj_list"],
        "Alg4+DRO": evaluation_results["bagging_alg4_DRO_obj_list"]
    }
    plot_CI_many_methods(list_obj, np.log2(sample_number))

    parameters = {
        "seed": seed,
        "r": r,
        "c": c,
        "q": q,
        "sample_args": sample_args,
        "B": B,
        "k": k,
        "B12": B12,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "varepsilon": varepsilon,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "large_number_sample": large_number_sample,
        "eval_time": eval_time
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f)
