from utils.SSKP_functions import comparison_twoPhase, evaluation_twoPhase
from utils.plotting import plot_twoPhase, plot_CI_twoPhase
import time
import numpy as np
import json


if __name__ == "__main__":
    # r = [3.2701236422941093, 3.3207149493214994, 3.556858029428708]
    # c, q = 3.7856629820554946, 1.7096129150007453
    # sample_args = {
    #         'type': 'pareto',
    #         'params': [2.0033248484659976, 1.9462659915572313, 2.0148555044660448]
    #     }

    # r = [
    #     2.68298539,
    #     3.81716309,
    #     4.60084485
    # ]
    # c = 3.3545021076444534
    # q = 2.4292864952500386

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
        # 1.94402027,
        # 2.18567363,
        # 1.91460062
        1.966,
        2.21,
        1.89
        ]}

    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200]
    k_list = [0.05, 0.1, 0.2, 0.4, 2, 10, 20, 50]
    B12_list = [(20,200)]
    epsilon = 0.005
    tolerance = 0.005
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(7, 13)])
    large_number_sample = 500000
    eval_time = 10

    # testing parameters
    # B_list = [2,3]
    # k_list = [0.1, 10]
    # B12_list = [(2,3),(3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.001
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])
    # large_number_sample = 100000
    # eval_time = 2

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, r, c, q)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, eval_time, rng_sample, sample_args, r, c, q)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list)

    parameters = {
        "seed": seed,
        "r": r,
        "c": c,
        "q": q,
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "large_number_sample": large_number_sample,
        "eval_time": eval_time
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f)




