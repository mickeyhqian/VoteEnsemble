import numpy as np
import json
import time
from utils.SSKP_functions import comparison_final, evaluation_final, evaluation_parallel
from utils.plotting import plot_final, plot_CI_final


if __name__ == "__main__":
    r = [4.274190740856761, 3.4532790823631223, 2.553734401951873]
    c, q = 3.7620908693653003, 1.3802150703969693
    # sample_args = {
    #         'type': 'pareto',
    #         'params': [2.0033248484659976, 1.9462659915572313, 2.0148555044660448]
    #     }
    sample_args = {
            'type': 'normal',
            'params': [[1.8432260340346258, 1.6550839716608632, 1.786452232195217], [0.9519161509875573, 0.5055798192930815, 1.1223363284198673]]
        }
    seed = 2024
    
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    # B_list = [100, 200, 400]
    # k_list = [0.05, 0.1, 10, 50]
    # number_of_iterations = 50
    # sample_number = np.array([2**i for i in range(6, 15)])
    # large_number_sample = 1000000

    # parameters for testing
    B_list = [2,3]
    k_list = [0.1, 2]
    number_of_iterations = 2
    sample_number = np.array([2**i for i in range(6, 8)])
    large_number_sample = 1000000
    eval_time = 10

    tic = time.time()
    SAA_list, bagging_list = comparison_final(B_list, k_list, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, r, c, q)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_list": bagging_list}, f)

    # SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg = evaluation_final(SAA_list, bagging_list, large_number_sample, rng_sample, sample_args, r, c,q)
    SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg = evaluation_parallel(SAA_list, bagging_list, large_number_sample, eval_time, rng_sample, sample_args, r, c, q)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_obj_list": bagging_obj_list, "bagging_obj_avg": bagging_obj_avg}, f)

    print(f"Total time: {time.time()-tic}")

    plot_final(SAA_obj_avg, bagging_obj_avg, np.log2(sample_number), B_list, k_list)
    plot_CI_final(SAA_obj_list, bagging_obj_list, np.log2(sample_number), B_list, k_list)

    parameters = {
        "seed": seed,
        "r": r,
        "c": c,
        "q": q,
        "sample_args": sample_args,
        "B_list": B_list,
        "k_list": k_list,
        "number_of_iterations": number_of_iterations,
        "sample_number": [int(x) for x in sample_number],
        "large_number_sample": large_number_sample
    }
    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)
    