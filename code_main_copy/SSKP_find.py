import time
import numpy as np
import matplotlib.pyplot as plt
import json

from utils.SSKP_functions import comparison_final, evaluation_final, evaluation_parallel

def find_parameters(m, B_list, k_list, number_of_iterations, sample_number, large_number_sample, rng, sample_type):
    # currently only supports len(B_list) = 1 and len(k_list) = 2
    parameters= []
    results_SAA = []
    results_bagging = []
    for iter in range(30):
        
        r = np.random.uniform(2.5, 5, size=m)
        r = [float(x) for x in r]
        c = float(np.random.uniform(2, 5, size=1)[0])
        q = float(np.random.uniform(0, 3, size=1)[0])

        if sample_type == 'pareto':
            params = np.random.uniform(1.7, 2.3, size=m)
            params = [float(x) for x in params]
        elif sample_type == 'normal':
            mean = np.random.uniform(1.5, 2.8, size=m)
            mean = [float(x) for x in mean]
            std = np.random.uniform(0.5, 1.8, size=m)
            std = [float(x) for x in std]
            params = [mean, std]
        
        sample_args = {
            'type': sample_type,
            'params': params
        }
        eval_time = 10

        tic = time.time()
        # since it is only used to find good parameters, the seed is not important
        SAA_list, bagging_list = comparison_final(B_list, k_list, number_of_iterations, sample_number, rng, rng, sample_args, r, c, q)
        # _, SAA_obj_avg, _, bagging_obj_avg = evaluation_final(SAA_list, bagging_list, large_number_sample, rng, sample_args, r, c, q)
        _, SAA_obj_avg, _, bagging_obj_avg = evaluation_parallel(SAA_list, bagging_list, large_number_sample, eval_time, rng, sample_args, r, c, q)
        print(f"Iteration {iter}, time: {time.time()-tic}")

        if all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][0])) and all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][1])):
            continue
        else:
            name = [iter, [r,c,q], params]
            plot_params(SAA_obj_avg, bagging_obj_avg, np.log2(sample_number), B_list, k_list, name)
            parameters.append([r, c, q, sample_args])
            results_SAA.append(SAA_obj_avg)
            results_bagging.append(bagging_obj_avg[0])
        
    return parameters, results_SAA, results_bagging

def plot_params(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list, name):
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
    fig_name  = str(name[0]) + '_' + str(name[1]) + '_' + str(name[2]) + ".png"
    plt.savefig(fig_name)
    # plt.show()
    return

        
if __name__ == "__main__":
    rng = np.random.default_rng(seed=2024)
    m = 3
    B_list = [200]
    k_list = [0.1,50]
    number_of_iterations = 30
    sample_number = np.array([2**i for i in range(6, 14)])
    large_number_sample = 1000000

    parameters, results_SAA, results_bagging = find_parameters(m, B_list, k_list, number_of_iterations, sample_number, large_number_sample, rng, 'normal')
    results = {'parameters': parameters, 'results_SAA': results_SAA, 'results_bagging': results_bagging}
    
    with open('results_normal.json', 'w') as f: 
        json.dump(results, f)

