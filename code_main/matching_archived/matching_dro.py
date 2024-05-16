import time
import json
import numpy as np
from utils.matching_functions import matching_obj_optimal, generateW, comparison_DRO, evaluation_DRO
from utils.plotting import plot_droComparison, plot_CI_droComparison, plot_optGap_droComparison

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
    
    # {
    #     (1, 2): 2.0,
    #     (1, 3): 2.0,
    #     (1, 4): 2.0,
    #     (1, 5): 1.6019243589320413,
    #     (1, 6): 1.516278165129409,
    #     (1, 7): 2.4946025818345543,
    #     (1, 8): 2.093167920562073,
    #     (2, 3): 2.0,
    #     (2, 4): 2.0,
    #     (2, 5): 1.5012536302329522,
    #     (2, 6): 2.3332096916501923,
    #     (2, 7): 2.2531359152024275,
    #     (2, 8): 1.5927195096204012,
    #     (3, 4): 2.0,
    #     (3, 5): 2.425508453846356,
    #     (3, 6): 1.761602924067154,
    #     (3, 7): 1.7531742944021391,
    #     (3, 8): 1.6103383846052868,
    #     (4, 5): 1.8423577681883925,
    #     (4, 6): 1.8279567487730057,
    #     (4, 7): 1.5566285841802934,
    #     (4, 8): 1.6673304038907535,
    #     (5, 6): 2,
    #     (5, 7): 2,
    #     (5, 8): 2,
    #     (6, 7): 2,
    #     (6, 8): 2,
    #     (7, 8): 2
    #     }
    

    sample_args = {
        "type" : "sym_pareto",
        "params": [2,2,2,2,2,2]
        # "params": [1.9,1.9,1.9,1.9,1.9,1.9]
    }

    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    # B_list = [200]
    # k_list = [10]
    # B12_list = [(20,200)]
    # epsilon = "dynamic"
    # tolerance = 0.005
    # varepsilon_list = [2**i for i in range(-6,4)] 
    # # varepsilon_list =  [10, 20, 50, 100, 1000]
    # number_of_iterations = 50
    # sample_number = np.array([2**i for i in range(7, 11)])

    # testing parameters
    # B_list = [2,3]
    # k_list = [10]
    # B12_list = [(2,3),(3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.001
    # varepsilon_list = [2**i for i in range(-6,-4)]
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    # tic = time.time()
    # SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_DRO(B_list, k_list, B12_list, epsilon, tolerance, varepsilon_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, N, w.copy())
    # with open("solution_lists.json", "w") as f:
    #     json.dump({"SAA_list": SAA_list, "dro_wasserstein_list": dro_wasserstein_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    # SAA_obj_list, SAA_obj_avg, dro_wasserstein_obj_list, dro_wasserstein_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w.copy())
    # with open("obj_lists.json", "w") as f:
    #     json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "dro_wasserstein_obj_list": dro_wasserstein_obj_list, "dro_wasserstein_obj_avg": dro_wasserstein_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    # print(f"Total time: {time.time()-tic}")

    # plot_droComparison(SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)
    # plot_CI_droComparison(SAA_obj_list, dro_wasserstein_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

    obj_opt, _ = matching_obj_optimal(sample_args, N, w)
    print(obj_opt)
    # plot_optGap_droComparison(obj_opt, "max", SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list, varepsilon_list)

    # parameters = {
    #     "seed": seed,
    #     "N": N,
    #     "w": {str(key): value for key, value in w.items()},
    #     "sample_args": sample_args,
    #     "B_list": B_list,
    #     "k_list": k_list,
    #     "B12_list": B12_list,
    #     "epsilon": epsilon,
    #     "tolerance": tolerance,
    #     "varepsilon_list": varepsilon_list,
    #     "number_of_iterations": number_of_iterations,
    #     "sample_number": sample_number.tolist(),
    #     "obj_opt": obj_opt
    # }

    # with open("parameters.json", "w") as f:
    #     json.dump(parameters, f)

