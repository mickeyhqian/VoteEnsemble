import time
import json
import numpy as np
from utils.LP_functions_full_random import comparison_epsilon, evaluation_epsilon, LP_obj_optimal, get_pareto_params
from utils.plotting import plot_epsilonComparison, plot_CI_epsilonComparison, plot_optGap_epsilonComparison


if __name__ == "__main__":
    seed = 2024
    N = 8
    w = {(1, 2): 3.0158429352664142,
 (1, 3): 3.043542817205786,
 (1, 4): 3.1669131932612897,
 (1, 5): 3.1467314427566855,
 (1, 6): 2.8936031297484033,
 (1, 7): 2.980858571209466,
 (1, 8): 3.1335646965676265,
 (2, 3): 3.0811886012220056,
 (2, 4): 3.043365053835512,
 (2, 5): 3.0638886123297677,
 (2, 6): 2.9347224075004426,
 (2, 7): 2.8717666772208967,
 (2, 8): 3.136450729603944,
 (3, 4): 2.8036979676766958,
 (3, 5): 3.019670801007701,
 (3, 6): 3.162854829949071,
 (3, 7): 2.8068588369297944,
 (3, 8): 2.825448429081741,
 (4, 5): 2.8621336956410626,
 (4, 6): 2.91777289015584,
 (4, 7): 2.8312545316964037,
 (4, 8): 3.130486589937699,
 (5, 6): 3.188920715768945,
 (5, 7): 2.8862035614453423,
 (5, 8): 3.101681642818335,
 (6, 7): 2.9378267306862464,
 (6, 8): 3.1919858495770885,
 (7, 8): 2.848287365067278}
    
#     {(1, 2): 2.0,
#  (1, 3): 1.909090909090909,
#  (1, 4): 1.8333333333333333,
#  (1, 5): 2.971165510976579,
#  (1, 6): 3.6698455195672723,
#  (1, 7): 2.945195467027686,
#  (1, 8): 2.807342645819148,
#  (2, 3): 1.7692307692307694,
#  (2, 4): 1.7142857142857144,
#  (2, 5): 2.822445662545111,
#  (2, 6): 3.7684430939757076,
#  (2, 7): 2.5074942266702553,
#  (2, 8): 3.394355765217963,
#  (3, 4): 1.6666666666666667,
#  (3, 5): 2.746521154099133,
#  (3, 6): 3.254754487739944,
#  (3, 7): 3.3000439818516814,
#  (3, 8): 3.425597587275234,
#  (4, 5): 2.899955766681089,
#  (4, 6): 2.703300518932373,
#  (4, 7): 2.9069686009794,
#  (4, 8): 3.00929316400101,
#  (5, 6): 3.0177290218662134,
#  (5, 7): 3.655745386610404,
#  (5, 8): 2.748966385209709,
#  (6, 7): 3.6437137707271217,
#  (6, 8): 3.4818655073739087,
#  (7, 8): 3.3735267160776634}

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

    params = get_pareto_params(N,w)

    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])

    sample_args = {
        "type" : "pareto",
        "params": params
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B = 200
    k_tuple = (10, 0.005)
    B12 = (20,200)
    epsilon_list = [0, 2**-3, 2**-1, 2, 8, "dynamic"]
    tolerance = 0.001
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(10, 16)])

    # test
    # B = 2
    # k_tuple = (10, 0.005)
    # B12 = (2,3)
    # epsilon_list = [0,"dynamic"]
    # tolerance = 0.1
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(6, 8)])

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, dyn_eps_alg3_list, dyn_eps_alg4_list = comparison_epsilon(B, k_tuple, B12, epsilon_list, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, N, w, A)
    with open("solution_lists_epsilon.json", "w") as f:
        json.dump({"SAA": SAA_list, "bagging_alg1": bagging_alg1_list, "bagging_alg3": bagging_alg3_list, "bagging_alg4": bagging_alg4_list, "dyn_eps_alg3": dyn_eps_alg3_list, "dyn_eps_alg4": dyn_eps_alg4_list}, f)

    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_epsilon(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w, A)
    with open("obj_lists_epsilon.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print("Elapsed time: ", time.time()-tic)

    plot_epsilonComparison(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B, k_tuple, B12, epsilon_list)
    plot_CI_epsilonComparison(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B, k_tuple, B12, epsilon_list)

    obj_opt, _ = LP_obj_optimal(N, w, A)
    plot_optGap_epsilonComparison(obj_opt, "max", SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B, k_tuple, B12, epsilon_list)

    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
        "A": A.tolist(),
        "sample_args": sample_args,
        "B": B,
        "k_tuple": k_tuple,
        "B12": B12,
        "epsilon_list": epsilon_list,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open("parameters_epsilon.json", "w") as f:
        json.dump(parameters, f, indent = 2)