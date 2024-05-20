from utils.network_functions import comparison_twoPhase, evaluation_twoPhase
from utils.plotting import plot_twoPhase, plot_CI_twoPhase
import time, json, numpy as np

if __name__ == "__main__":
    s, p, c, g = 3, 2, 3, 5

    C = np.array([0.84902017, 0.73141691])
    Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
                        [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
                        [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
                        [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
                        [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
                        [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
                        [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
                        [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
                        [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
                        [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
                        [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
                    [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    M = np.array([0.96766421, 0.5369463])
    H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
                    [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
                    [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])

    sample_args = {
        'type': 'pareto',
        'size': (s,c,g),
        'params': [1.90394177, 2.00408536]
    }

    seed = 2024
    rng_sample = np.random.default_rng(seed=seed)
    rng_alg = np.random.default_rng(seed=seed*2)
    
    B_list = [200]
    k_list = [0.1, 10]
    B12_list = [(20,200)]
    epsilon = "dynamic"
    tolerance = 0.005
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(7, 12)])
    large_number_sample = 100000
    eval_time = 10

    tic = time.time()
    SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list = comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, s, C, Q_sp, Q_pc, R, M, H)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_alg1_list": bagging_alg1_list, "bagging_alg3_list": bagging_alg3_list, "bagging_alg4_list": bagging_alg4_list}, f)
    
    SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg = evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, eval_time, rng_sample, sample_args, s, C, Q_sp, Q_pc, R, M, H)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_alg1_obj_list": bagging_alg1_obj_list, "bagging_alg1_obj_avg": bagging_alg1_obj_avg, "bagging_alg3_obj_list": bagging_alg3_obj_list, "bagging_alg3_obj_avg": bagging_alg3_obj_avg, "bagging_alg4_obj_list": bagging_alg4_obj_list, "bagging_alg4_obj_avg": bagging_alg4_obj_avg}, f)
    print(f"Total time: {time.time()-tic}")

    plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, np.log2(sample_number), B_list, k_list, B12_list)
    plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, np.log2(sample_number), B_list, k_list, B12_list)

    parameters = {
        'seed': seed,
        's': s,
        'p': p,
        'c': c,
        'g': g,
        'C': C.tolist(),
        'Q_sp': Q_sp.tolist(),
        'Q_pc': Q_pc.tolist(),
        'R': R.tolist(),
        'M': M.tolist(),
        'H': H.tolist(),
        'sample_args': sample_args,
        'B_list': B_list,
        'k_list': k_list,
        'B12_list': B12_list,
        'epsilon': epsilon,
        'tolerance': tolerance,
        'number_of_iterations': number_of_iterations,
        'sample_number': sample_number.tolist(),
        'large_number_sample': large_number_sample,
        'eval_time': eval_time
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)
