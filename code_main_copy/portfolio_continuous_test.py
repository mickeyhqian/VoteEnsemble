import time
import json
import numpy as np
from utils.portfolio_functions_continuous import comparison_test, evaluation_test, get_pareto_params
from utils.plotting import plot_test, plot_CI_test


if __name__ == "__main__":
    seed = 2024
    mu = np.random.uniform(1.2,1.95, 6)
    b = np.random.uniform(1.2, 1.6)

    params = get_pareto_params(mu)
    sample_args = {
        "type" : "pareto",
        "params": params
    }

    k_list = [(10,0.005), (20, 0.01), (50, 0.01)]
    B1_list = [100, 200, 400, 600]
    number_of_iterations = 20
    sample_number = np.array([2**i for i in range(7, 16, 2)])

    # test
    # k_list = [(10,0.005)]
    # B1_list = [2]
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(6, 8)])

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    name = str(mu[0])

    tic = time.time()
    SAA_list, bagging_alg3_list, bagging_alg3_trick_list = comparison_test(k_list, B1_list, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, mu, b)
    with open(name+"solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg3_trick_list": bagging_alg3_trick_list}, f)
    
    SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg3_trick_obj_list, bagging_alg3_trick_obj_avg = evaluation_test(SAA_list, bagging_alg3_list, bagging_alg3_trick_list, mu, b)
    with open(name+"obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg3_trick_obj_list": bagging_alg3_trick_obj_list, "bagging_alg3_trick_obj_avg": bagging_alg3_trick_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    plot_test(SAA_obj_avg, bagging_alg3_obj_avg, bagging_alg3_trick_obj_avg, sample_number, k_list, B1_list, name)
    plot_CI_test(SAA_obj_list, bagging_alg3_obj_list, bagging_alg3_trick_obj_list, sample_number, k_list, B1_list, name)

    parameters = {
        "seed": seed,
        "mu": mu.tolist(),
        "b": b,
        "sample_args": sample_args,
        "k_list": k_list,
        "B1_list": B1_list,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist()
    }

    with open(name+"parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)

