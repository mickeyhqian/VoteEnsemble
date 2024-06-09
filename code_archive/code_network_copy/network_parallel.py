import numpy as np
import time
import json
from network_functions import comparison_final, evaluation_final, plot_final, plot_CI_final, evaluation_parallel

if __name__ == "__main__":
    s, p, c, g = 3, 2, 3, 5

    # C = np.array([0.84902017, 0.73141691])
    # Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
    #                     [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
    #                     [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
    #                     [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
    #                     [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
    #                     [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    # Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
    #                     [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
    #                     [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
    #                     [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
    #                     [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
    #                     [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    # R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
    #                 [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    # M = np.array([0.96766421, 0.5369463])
    # H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
    #                 [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
    #                 [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])

    # sample_args = {
    #     'type': 'pareto',
    #     'size': (s,c,g),
    #     'params': [1.90394177, 2.00408536]
    # }

    C =  np.array([
    0.7919227499225773,
    0.4309762892487943
    ])
    Q_sp =  np.array([
    [
        [
        0.33959485,
        0.93963167,
        0.11099611,
        0.29228903,
        0.32671781
        ],
        [
        0.11183712,
        0.68568047,
        0.89458674,
        0.6290198,
        0.45712153
        ]
    ],
    [
        [
        0.30562386,
        0.25141794,
        0.52145227,
        0.93541722,
        0.09209385
        ],
        [
        0.32229624,
        0.78332703,
        0.97653641,
        0.18445241,
        0.03020424
        ]
    ],
    [
        [
        0.3414426,
        0.45228102,
        0.22978903,
        0.05338668,
        0.24009525
        ],
        [
        0.92622445,
        0.08852084,
        0.84160819,
        0.02508719,
        0.46203217
        ]
    ]
    ])
    Q_pc = np.array([
    [
        [
        0.08967017,
        0.08001494,
        0.29854161,
        0.6979279,
        0.12176636
        ],
        [
        0.96381038,
        0.1531089,
        0.38302768,
        0.61996695,
        0.90823239
        ],
        [
        0.70189534,
        0.39932218,
        0.55522219,
        0.00271483,
        0.89190262
        ]
    ],
    [
        [
        0.00499716,
        0.94445126,
        0.88466098,
        0.5649098,
        0.54066034
        ],
        [
        0.35775422,
        0.14473548,
        0.17228491,
        0.64621086,
        0.6801251
        ],
        [
        0.22298374,
        0.17262673,
        0.42564852,
        0.25968014,
        0.01474239
        ]
    ]
    ])
    R = np.array([
    [
        0.12818939278211294,
        0.5542344960524943,
        0.9854726132978918,
        0.43879198808177977,
        0.14723859888465474
    ],
    [
        0.5264218506399049,
        0.3839484660878939,
        0.15881886701443104,
        0.1522486534351991,
        0.7744995797298272
    ]
    ])
    M = np.array([
    0.578235507734427,
    0.7864622252392603
    ])
    H = np.array([
    [
        0.591587608870183,
        0.026957066402544805,
        0.9135846040334558,
        0.48745794285753385,
        0.27618031129853293
    ],
    [
        0.7757970175128,
        0.21574920436927703,
        0.7973105450979264,
        0.28312662250371745,
        0.8463864002304629
    ],
    [
        0.695585019276411,
        0.987994577307807,
        0.4064966522430422,
        0.19988585016960747,
        0.7756998416359008
    ]
    ])
    sample_args = {
    "type": "pareto",
    "size": [
        3,
        3,
        5
    ],
    "params": [
        1.902719726676329,
        1.909028679018095
    ]
    }

    seed = 2024
    rng = np.random.default_rng(seed=seed)

    # B_list = [100, 200, 400]
    # k_list = [0.05, 0.1, 10, 50]
    # number_of_iterations = 50
    # sample_number = np.array([2**i for i in range(6, 13)])
    # large_number_sample = 200000

    # parameters for testing
    B_list = [200, 400]
    k_list = [0.1, 0.2, 2, 10, 50]
    number_of_iterations = 50
    sample_number = np.array([2**i for i in range(6, 15)])
    large_number_sample = 100000
    eval_time = 10

    tic = time.time()
    SAA_list, bagging_list = comparison_final(B_list, k_list, number_of_iterations, sample_number, rng, sample_args, C, Q_sp, Q_pc, R, M, H)
    with open("solution_lists.json", "w") as f:
        json.dump({"SAA_list": SAA_list, "bagging_list": bagging_list}, f)

    # SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg = evaluation_final(SAA_list, bagging_list, large_number_sample, rng, sample_args, C, Q_sp, Q_pc, R, M, H)
    SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg = evaluation_parallel(SAA_list, bagging_list, large_number_sample, eval_time, rng, sample_args, C, Q_sp, Q_pc, R, M, H)
    with open("obj_lists.json", "w") as f:
        json.dump({"SAA_obj_list": SAA_obj_list, "SAA_obj_avg": SAA_obj_avg, "bagging_obj_list": bagging_obj_list, "bagging_obj_avg": bagging_obj_avg}, f)

    print(f"Total time: {time.time()-tic}")

    plot_final(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list)
    plot_CI_final(SAA_obj_list, bagging_obj_list, sample_number, B_list, k_list)

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
        'number_of_iterations': number_of_iterations,
        'sample_number': sample_number.tolist(),
        'large_number_sample': large_number_sample
        }
    
    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)







