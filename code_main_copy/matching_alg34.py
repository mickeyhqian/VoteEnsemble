import time
import json
import numpy as np
from utils.matching_functions import comparison_twoPhase, evaluation_twoPhase, matching_obj_optimal, generateW
from utils.plotting import plot_twoPhase, plot_CI_twoPhase, plot_optGap_twoPhase




if __name__ == "__main__":
    seed = 2024
    N = 8
    # w = {(1, 2): None,
    #     (1, 3): None,
    #     (1, 4): None,
    #     (1, 5): 1.9119257287734244,
    #     (1, 6): 1.6228257372263202,
    #     (1, 7): 1.7968885916853101,
    #     (1, 8): 2.313248853705918,
    #     (2, 3): None,
    #     (2, 4): None,
    #     (2, 5): 1.9362072829402424,
    #     (2, 6): 2.451003072707256,
    #     (2, 7): 2.398840628680899,
    #     (2, 8): 1.8209819889450793,
    #     (3, 4): None,
    #     (3, 5): 1.9978624908336502,
    #     (3, 6): 2.0670783259937506,
    #     (3, 7): 1.8577180932652504,
    #     (3, 8): 2.3907208610757693,
    #     (4, 5): 2.203256964324676,
    #     (4, 6): 2.422592793261833,
    #     (4, 7): 1.6985434453495578,
    #     (4, 8): 1.9787435233099542,
    #     (5, 6): 2,
    #     (5, 7): 2,
    #     (5, 8): 2,
    #     (6, 7): 2,
    #     (6, 8): 2,
    #     (7, 8): 2}

    w = {
    (1, 2): None,
    (1, 3): None,
    (1, 4): None,
    (1, 5): 2.003372258673751,
    (1, 6): 2.0115463019831132,
    (1, 7): 1.9542517342707688,
    (1, 8): 2.0141317529282925,
    (2, 3): None,
    (2, 4): None,
    (2, 5): 2.046239713130396,
    (2, 6): 2.0282889751814497,
    (2, 7): 2.010090258614592,
    (2, 8): 2.036033009209721,
    (3, 4): None,
    (3, 5): 2.049760410264224,
    (3, 6): 1.9684908395780378,
    (3, 7): 2.019231010951654,
    (3, 8): 1.9982793554639287,
    (4, 5): 2.0102514380473737,
    (4, 6): 1.9828975541772373,
    (4, 7): 2.022110609409712,
    (4, 8): 1.9572793076811055,
    (5, 6): 1.9560358475155626,
    (5, 7): 2.006760926782303,
    (5, 8): 1.976932108263151,
    (6, 7): 2.019051215736671,
    (6, 8): 1.9964954091971618,
    (7, 8): 1.9603393228495931
    }

    # w = generateW(N,"random")
    # w = generateW(N)
    sample_args = {
        # "type" : "pareto",
        "type" : "normal",
        # "params": [2,2,2,2,2,2]
        # "params": np.random.uniform(1.95,2.05,6).tolist()
        # "params": [[2,2,2,2,2,2], np.random.uniform(1,1.5,6).tolist()]
        "params": [
            [1.9790712457872344, 1.9646657054823478, 1.9845813076676622, 2.006175658708957, 1.962250459541956, 1.9699279489289765],
            np.random.uniform(1,1.5,6).tolist()
        ]
    }
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)

    B_list = [200]
    k_list = [0.1, 10]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 0.005
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(7, 12)])

    # test
    # B_list = [2,3]
    # k_list = [0.1, 10]
    # B12_list = [(2,3), (3,3)]
    # epsilon = "dynamic"
    # tolerance = 0.005
    # number_of_iterations = 2
    # sample_number = np.array([2**i for i in range(7, 9)])

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, N, w.copy())
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)

    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, N, w.copy())
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    # plot required graphs
    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list)

    obj_opt, _ = matching_obj_optimal(sample_args, N, w)
    plot_optGap_twoPhase(obj_opt, "max", SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list)

    parameters = {
        "seed": seed,
        "N": N,
        "w": {str(key): value for key, value in w.items()},
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

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=1)