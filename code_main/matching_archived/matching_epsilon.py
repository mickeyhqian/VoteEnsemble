import time
import json
import numpy as np
from utils.matching_functions import comparison_epsilon, evaluation_epsilon, matching_obj_optimal, generateW
from utils.plotting import plot_epsilonComparison, plot_CI_epsilonComparison, plot_optGap_epsilonComparison

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

    # test
    B = 200
    k = 10
    B12 = (20,200)
    epsilon_list = [0] + np.array([2**i for i in range(-9,2)]).tolist() + ['dynamic']
    tolerance = 0.001
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(6, 13)])

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, dyn_eps_alg3_list, dyn_eps_alg4_list = comparison_epsilon(B, k, B12, epsilon_list, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, N, w.copy())
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA": SAA_list, "bagging_alg1": bagging_alg1_list, "bagging_alg3": bagging_alg3_list, "bagging_alg4": bagging_alg4_list, "dyn_eps_alg3": dyn_eps_alg3_list, "dyn_eps_alg4": dyn_eps_alg4_list}, f)
    
    dyn_eps_alg3_list_avg = np.mean(dyn_eps_alg3_list, axis=1).tolist()
    dyn_eps_alg4_list_avg = np.mean(dyn_eps_alg4_list, axis=1).tolist()

    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_epsilon(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w.copy())
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    
    print(f"Total time: {time.time()-tic}")

    plot_epsilonComparison(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B, k, B12, epsilon_list)
    plot_CI_epsilonComparison(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B, k, B12, epsilon_list)

    obj_opt, _ = matching_obj_optimal(sample_args, N, w)
    plot_optGap_epsilonComparison(obj_opt, "max", SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B, k, B12, epsilon_list)

    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
        "sample_args": sample_args,
        "B": B,
        "k": k,
        "B12": B12,
        "epsilon_list": epsilon_list,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist(),
        "obj_opt": obj_opt
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent = 1)


    
