import time
import json
import numpy as np
from utils.LR_functions import comparison_twoPhase, evaluation_twoPhase
from utils.plotting import plot_twoPhase, plot_CI_twoPhase, plot_twoPhase_cdf

if __name__ == "__main__":
    seed = 666
    rng_problem = np.random.default_rng(seed=seed-1)
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    meanX = rng_problem.uniform(1.1, 1.9, 10)
    beta_true = rng_problem.uniform(1, 20, 10)
    noise = 2.1
    sample_args = {
        "meanX": meanX, # mean values of the x_vector
        "params": [item/(item-1) for item in meanX], # list of pareto shapes
        "beta_true": beta_true, # underlying true beta vector
        "noise": noise # noise shape
    }

    B_list = []
    k_list = [(30, 0.5)]
    k2_tuple = (30, 0.005)
    B12_list = [(30, 200)]
    epsilon = 'dynamic'
    tolerance = 1e-3
    number_of_iterations = 200
    sample_number = np.array([2**i for i in range(10, 17)])

    # test
    # B_list = []
    # k_list = [(10,0.01)]
    # B12_list = [(2,3), (3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.1
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    name = str(beta_true[0])

    tic = time.time()
    SAA_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number, rng_sample, rng_alg, sample_args, k2_tuple = k2_tuple)
    with open(name + "solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg3_list, bagging_alg4_list, sample_args)
    with open(name + "obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    bagging_alg1_obj_avg = None
    bagging_alg1_obj_list = None
    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, name = name, skip_alg1=True)
    plot_twoPhase_cdf(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list, name = name, skip_alg1=True)
    # plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, name= name, skip_alg1=True)

    parameters = {
        "seed": seed,
        "meanX": meanX.tolist(),
        "beta_true": beta_true.tolist(),
        "noise": noise,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist()
    }

    with open(name + "parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)