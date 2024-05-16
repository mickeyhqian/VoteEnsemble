import time
import json
import numpy as np
from utils.portfolio_functions_continuous import comparison_twoPhase, evaluation_twoPhase, get_pareto_params
from utils.plotting import plot_twoPhase, plot_CI_twoPhase

if __name__ == "__main__":
    seed = 2024
    # mu = np.random.uniform(1.2,1.8, 6)
    # b = np.random.uniform(1.2, 1.6)

    mu = np.array([
    1.298022168012522,
    1.5139145944126466,
    1.74027529860629,
    1.6597781125096558,
    1.6225302291466706,
    1.6631624129206548
  ])
    b = 1.357202771189245

    params = get_pareto_params(mu)
    sample_args = {
        "type" : "pareto",
        "params": params
    }

    B_list = []
    k_list = [(10,0.005), (20, 0.01), (50, 0.05)]
    B12_list = [(200,200)]
    epsilon = 'dynamic'
    tolerance = 0.001
    number_of_iterations = 20
    sample_number = np.array([2**i for i in range(7, 16, 2)])

    # test
    # B_list = []
    # k_list = [(10,0.005)]
    # B12_list = [(2,3)]
    # epsilon = 'dynamic'
    # tolerance = 0.005
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(6, 8)])

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    name = str(mu[0])

    tic = time.time()
    SAA_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, mu, b)
    with open(name+"solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg3_list, bagging_alg4_list, mu, b)
    with open(name+"obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    bagging_alg1_obj_avg = None
    bagging_alg1_obj_list = None
    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, name = name, skip_alg1=True)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, name= name, skip_alg1=True)

    parameters = {
        "seed": seed,
        "mu": mu.tolist(),
        "b": b,
        "sample_args": sample_args,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist()
    }

    with open(name + "parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)


