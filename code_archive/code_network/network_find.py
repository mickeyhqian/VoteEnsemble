import numpy as np
import time
import json
from network_functions import comparison_final, evaluation_final, evaluation_parallel
import matplotlib.pyplot as plt


def plot_params(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list, name):
    sample_number = np.log2(sample_number)
    _, ax = plt.subplots()
    # ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            ax.plot(sample_number, [bagging_obj_avg[ind1][ind2][i] - SAA_obj_avg[i] for i in range(len(sample_number))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'B={B_list[ind1]}, k={k_list[ind2]}')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective difference', size = 20)
    ax.legend()
    fig_name  = str(name[0]) + '_' + str(name[1]) + ".png"
    plt.savefig(fig_name)
    # plt.show()
    return

def find_parameters(s, p, c, g, B_list, k_list, number_of_iterations,sample_number, large_number_sample, rng):
    # currently only supports len(B_list) = 1 and len(k_list) = 1
    for iter in range(400):
        C = np.random.rand(p)
        Q_sp = np.random.rand(s, p, g)
        Q_pc = np.random.rand(p, c, g)
        R = np.random.rand(p, g)
        M = np.random.rand(p)
        H = np.random.rand(c, g)

        param = np.random.uniform(1.6,2)
        params = [param, param + np.random.uniform(0,0.4)]

        sample_args = {
            'type': 'pareto',
            'size': (s,c,g),
            'params': params
        }

        eval_time = 2
        tic = time.time()
        SAA_list, bagging_list = comparison_final(B_list, k_list, number_of_iterations, sample_number, rng, sample_args, C, Q_sp, Q_pc, R, M, H)
        _, SAA_obj_avg, _, bagging_obj_avg = evaluation_parallel(SAA_list, bagging_list, large_number_sample, eval_time, rng, sample_args, C, Q_sp, Q_pc, R, M, H)
        print(f"Iteration {iter+1} took {time.time()-tic} seconds")

        # if all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][0])) and all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][1])):
        name = [iter+1, params]
        plot_params(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list, name)
        if all(x <= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][0])):
            continue
        else:
            parameters = {
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
            }
            file_name = str(iter+1) + "_parameters.json"
            with open(file_name, "w") as f:
                json.dump(parameters, f, indent=2)
    return 

if __name__ == "__main__":
    rng = np.random.default_rng(seed=2024)
    
    s, p, c, g = 3, 2, 3, 5
    B_list = [100]
    k_list = [0.1]
    number_of_iterations = 1
    sample_number = np.array([2**i for i in range(6, 11)])
    large_number_sample = 50000

    find_parameters(s, p, c, g, B_list, k_list, number_of_iterations, sample_number, large_number_sample, rng)

            
        


