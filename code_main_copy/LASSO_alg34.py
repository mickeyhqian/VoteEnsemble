import time
import json
import numpy as np
from utils.LASSO_function import comparison_twoPhase, evaluation_twoPhase, problem_initialization, get_expected_loss_dict
from utils.plotting import plot_twoPhase, plot_CI_twoPhase



if __name__ == "__main__":
    seed = 2024
    
    # rng_problem = np.random.default_rng(seed=seed-1)
    # rng_train = np.random.default_rng(seed=seed*3)

    # meanX = rng_problem.uniform(1.1, 1.9,3000)
    # beta_true = rng_problem.uniform(1,20,3000)
    # noise = 3
    # sample_args = {
    #     "meanX": meanX, # mean values of the x_vector
    #     "params": [item/(item-1) for item in meanX], # list of pareto shapes
    #     "beta_true": beta_true, # underlying true beta vector
    #     "noise": noise # noise shape
    # }

    # lambda_list = [i for i in range(1,11)]
    # sample_number_train = 1000

    # beta_dict = problem_initialization(lambda_list, sample_number_train, rng_train, sample_args)
    # expected_loss_dict = get_expected_loss_dict(sample_args, beta_dict)

    # problems = {
    #     "meanX": meanX.tolist(),
    #     "beta_true": beta_true.tolist(),
    #     "noise": noise,
    #     "lambda_list": lambda_list,
    #     "expected_loss_dict": expected_loss_dict,
    #     "sample_number_train": sample_number_train
    # }

    # with open("problems.json", "w") as f:
    #     json.dump(problems, f, indent = 2)

    # with open("beta_dict.json", "w") as f:
    #     # create a new dictionary for beta_dict, convert numpy arrays to list
    #     beta_dict_json = {}
    #     for key, value in beta_dict.items():
    #         beta_dict_json[key] = value.tolist()
    #     json.dump(beta_dict_json, f, indent = 2)

    with open("_LASSO_beta_dict.json", "r") as f:
        beta_dict_json = json.load(f)
    beta_dict = {}
    for key, value in beta_dict_json.items():
        beta_dict[eval(key)] = np.array(value)
    
    with open("_LASSO_problems.json", "r") as f:
        problems = json.load(f)
    meanX = np.array(problems["meanX"])
    beta_true = np.array(problems["beta_true"])
    noise = problems["noise"]
    lambda_list = problems["lambda_list"]
    expected_loss_dict = {int(key): value for key, value in problems["expected_loss_dict"].items()}
    
    sample_args = {
        "meanX": meanX, # mean values of the x_vector
        "params": [item/(item-1) for item in meanX], # list of pareto shapes
        "beta_true": beta_true, # underlying true beta vector
        "noise": noise # noise shape
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)


    B_list = [200]
    k_list = [(10,0.005)]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 1
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(5, 12)])

    # test
    # B_list = [2,3]
    # k_list = [(10,0.01)]
    # B12_list = [(2,3), (3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.1
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number, rng_sample, rng_alg, sample_args, beta_dict)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    # with open("solution_lists.json", "r") as f:
    #     solution_lists = json.load(f)
    # SAA_list = solution_lists["SAA_list"]
    # bagging_alg1_list = solution_lists["bagging_alg1_list"]
    # bagging_alg3_list = solution_lists["bagging_alg3_list"]
    # bagging_alg4_list = solution_lists["bagging_alg4_list"]

    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, expected_loss_dict)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list)

    parameters = {
        "seed": seed,
        "B_list": B_list,
        "k_list": k_list,
        "B12_list": B12_list,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "number_of_iterations": number_of_iterations,
        "sample_number": sample_number.tolist()
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)

        
    
