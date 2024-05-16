import numpy as np
import time
import json
from utils.portfolio_functions import comparison_twoPhase, evaluation_twoPhase
from utils.plotting import plot_params

def find_parameters(m, B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number, large_number_sample):
    for iter in range(100):
        seed = 2024
        rng_sample = np.random.default_rng(seed=seed)
        rng_alg = np.random.default_rng(seed=seed*2)
        mu = np.random.uniform(1,5, m)
        sample_args = {
            "type" : "sym_pareto",
            "params": mu.tolist()
        }

        p = np.random.uniform(0, 2, m)
        b = np.random.uniform(1, 3)
        alpha = 0.95

        tic = time.time()
        SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, p, mu, b, alpha)
        _, SAA_obj_avg, _, bagging_alg1_obj_avg, _, bagging_alg3_obj_avg, _, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, rng_sample, sample_args, p, mu, b, alpha)
        print(f"Iteration {iter+1} took {time.time()-tic} seconds")

        name = str([iter+1, mu])
        # plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, name)
        plot_params(SAA_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B12_list, k_list, name)
        
        parameters = {
            "m": m,
            "mu": mu.tolist(),
            "p": p.tolist(),
            "b": b,
            "alpha": alpha,
            "sample_args": sample_args,
            "B_list": B_list,
            "k_list": k_list,
            "B12_list": B12_list,
            "epsilon": epsilon,
            "tolerance": tolerance,
            "number_of_iterations": number_of_iterations,
            "sample_number": sample_number.tolist(),
            "large_number_sample": large_number_sample
        }

        with open(f"{name}_parameters.json", "w") as f:
            json.dump(parameters, f, indent=2)

    return

if __name__ == "__main__":
    m = 6
    
    B_list = [1]
    k_list = [0.1]
    B12_list = [(20,100)]
    epsilon = 0.01
    tolerance = 0.005
    number_of_iterations = 3
    sample_number = np.array([2**i for i in range(7, 12)])
    large_number_sample = 1000000

    find_parameters(m, B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, large_number_sample)
