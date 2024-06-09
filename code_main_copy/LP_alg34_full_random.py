import time
import json
import numpy as np
from utils.LP_functions_full_random import comparison_twoPhase, evaluation_twoPhase, LP_obj_optimal, get_pareto_params
from utils.plotting import plot_twoPhase, plot_CI_twoPhase, plot_optGap_twoPhase

if __name__ == "__main__":
    seed = 2024
    N = 8
    w =  {(1, 2): 2.093667789973577,
 (1, 3): 2.0397679467629555,
 (1, 4): 1.9623921908985666,
 (1, 5): 1.985881773116472,
 (1, 6): 2.0146552711627255,
 (1, 7): 2.052269735810645,
 (1, 8): 1.9512246691506894,
 (2, 3): 1.9946342543842692,
 (2, 4): 1.902223187194079,
 (2, 5): 2.030302650367223,
 (2, 6): 2.027109460425652,
 (2, 7): 2.041204077930563,
 (2, 8): 2.0949420264651897,
 (3, 4): 1.9229480405442636,
 (3, 5): 1.9659767694352177,
 (3, 6): 2.001319833583405,
 (3, 7): 1.9414943589648475,
 (3, 8): 2.0760931415435633,
 (4, 5): 2.014052335560319,
 (4, 6): 1.9963225944759295,
 (4, 7): 2.0495209285888514,
 (4, 8): 1.9157493055577421,
 (5, 6): 2.0478500482464246,
 (5, 7): 1.92069555926702,
 (5, 8): 1.9978157241409709,
 (6, 7): 1.9779129653049157,
 (6, 8): 1.9820794352442952,
 (7, 8): 1.9531038713889752}
    
    for key in w.keys():
        w[key] = np.random.uniform(2.8, 3.2)

    
    # {(1, 2): 3,
    # (1, 3): 3,
    # (1, 4): 3,
    # (1, 5): 3.5,
    # (1, 6): 3.5,
    # (1, 7): 3.5,
    # (1, 8): 3.5,
    # (2, 3): 3,
    # (2, 4): 3,
    # (2, 5): 3.5,
    # (2, 6): 3.5,
    # (2, 7): 3.5,
    # (2, 8): 3.5,
    # (3, 4): 3,
    # (3, 5): 3.5,
    # (3, 6): 3.5,
    # (3, 7): 3.5,
    # (3, 8): 3.5,
    # (4, 5): 3.5,
    # (4, 6): 3.5,
    # (4, 7): 3.5,
    # (4, 8): 3.5,
    # (5, 6): 3.5,
    # (5, 7): 3.5,
    # (5, 8): 3.5,
    # (6, 7): 3.5,
    # (6, 8): 3.5,
    # (7, 8): 3.5}
    
    # {(1, 2): 2.5,
    # (1, 3): 2.5,
    # (1, 4): 2.5,
    # (1, 5): 3,
    # (1, 6): 3,
    # (1, 7): 3,
    # (1, 8): 3,
    # (2, 3): 2.5,
    # (2, 4): 2.5,
    # (2, 5): 3,
    # (2, 6): 3,
    # (2, 7): 3,
    # (2, 8): 3,
    # (3, 4): 2.5,
    # (3, 5): 3,
    # (3, 6): 3,
    # (3, 7): 3,
    # (3, 8): 3,
    # (4, 5): 3,
    # (4, 6): 3,
    # (4, 7): 3,
    # (4, 8): 3,
    # (5, 6): 3,
    # (5, 7): 3,
    # (5, 8): 3,
    # (6, 7): 3,
    # (6, 8): 3,
    # (7, 8): 3}

#     {(1, 2): 2,
# (1, 3): 2,
# (1, 4): 2,
# (1, 5): 2.5,
# (1, 6): 2.5,
# (1, 7): 2.5,
# (1, 8): 2.5,
# (2, 3): 2,
# (2, 4): 2,
# (2, 5): 2.5,
# (2, 6): 2.5,
# (2, 7): 2.5,
# (2, 8): 2.5,
# (3, 4): 2,
# (3, 5): 2.5,
# (3, 6): 2.5,
# (3, 7): 2.5,
# (3, 8): 2.5,
# (4, 5): 2.5,
# (4, 6): 2.5,
# (4, 7): 2.5,
# (4, 8): 2.5,
# (5, 6): 2.5,
# (5, 7): 2.5,
# (5, 8): 2.5,
# (6, 7): 2.5,
# (6, 8): 2.5,
# (7, 8): 2.5}
    
    # w = {(1, 2): 10,
    # (1, 3): 10,
    # (1, 4): 10,
    # (1, 5): 11,
    # (1, 6): 11,
    # (1, 7): 11,
    # (1, 8): 11,
    # (2, 3): 10,
    # (2, 4): 10,
    # (2, 5): 11,
    # (2, 6): 11,
    # (2, 7): 11,
    # (2, 8): 11,
    # (3, 4): 10,
    # (3, 5): 11,
    # (3, 6): 11,
    # (3, 7): 11,
    # (3, 8): 11,
    # (4, 5): 11,
    # (4, 6): 11,
    # (4, 7): 11,
    # (4, 8): 11,
    # (5, 6): 11,
    # (5, 7): 11,
    # (5, 8): 11,
    # (6, 7): 11,
    # (6, 8): 11,
    # (7, 8): 11}

    # N = 6
    # w = {(1, 2): 4,
    #      (1, 3): 4,
    #      (1, 4): 5,
    #     (1, 5): 5,
    #     (1, 6): 5,
    #     (2, 3): 4,
    #     (2, 4): 5,
    #     (2, 5): 5,
    #     (2, 6): 5,
    #     (3, 4): 5,
    #     (3, 5): 5,
    #     (3, 6): 5,
    #     (4, 5): 5,
    #     (4, 6): 5,
    #     (5, 6): 5}

    params = get_pareto_params(N,w)

    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    # A = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])

    sample_args = {
        "type" : "pareto",
        "params": params
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200]
    # k_list = [(10, 0.002), (10,0.005), (10, 0.01)]
    k_list = [(10, 0.005)]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 0.001
    number_of_iterations = 20
    sample_number = np.array([2**i for i in range(8, 15)])
    # sample_number = np.array([1000, 5000, 10000, 20000])
    
    # test
    # B_list = [2,3]
    # k_list = [0.1, 10]
    # B12_list = [(2,3), (3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.005
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    name = str(w[(1, 2)])

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, N, w, A)
    with open(name + "solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)

    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w, A)
    with open(name + "obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    # plot required graphs
    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, name = name)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, name = name)

    obj_opt, _ = LP_obj_optimal(N, w, A)
    plot_optGap_twoPhase(obj_opt, "max", SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, name = name)

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
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open(name + "parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)