import numpy as np
from scipy import stats
from scipy.integrate import quad
from ParallelSolve import solveNewsVendorSAA, majority_vote
import matplotlib.pyplot as plt
import time
import json

def genParetoDemand(n, rng, paretoShape):
    return stats.lomax.rvs(paretoShape, size=n, random_state=rng)

def paretoOptima(price, cost, paretoShape):
    # directly solve the newvendor
    q = (price - cost) / price
    opt = stats.lomax.ppf(q, paretoShape)
    return np.ceil(opt) # take ceiling of the optimal order quantity

def paretoObj(x, paretoShape, price, cost):
    # computes the objective function for the newvendor
    def func(z):
        # density of pareto distribution
        return z * paretoShape / (1 + z) ** (paretoShape + 1)
    integral = quad(func, 0, x)[0]
    return price * (integral + x * (1 - stats.lomax.cdf(x, paretoShape))) - cost * x

def comparison_final(B_list,k_list,number_of_iterations,sample_number,rng, sample_args, *prob_args):
    # a unified function that compare SAA with different configurations of Bagging
    # prob_args: price, cost
    # sample_args: paretoShapes
    SAA_list = []
    bagging_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))] 
    for n in sample_number:
        SAA_intermediate = []
        bagging_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        for _ in range(number_of_iterations):
            tic = time.time()
            sample_n = genParetoDemand(n, rng, sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, solveNewsVendorSAA, rng, *prob_args)
            SAA_intermediate.append(SAA)
            print(f"Sample size {n}, iteration {_}, SAA time: {time.time()-tic}")

            for ind1, B in enumerate(B_list):
                for ind2, k in enumerate(k_list):
                    tic = time.time()
                    if k < 1:
                        bagging, _ = majority_vote(sample_n, B, int(n*k), solveNewsVendorSAA, rng, *prob_args)
                    else:
                        bagging, _ = majority_vote(sample_n, B, k, solveNewsVendorSAA, rng, *prob_args)
                    bagging_intermediate[ind1][ind2].append(bagging)
                    print(f"Sample size {n}, iteration {_}, B={B}, k={k}, Bagging time: {time.time()-tic}")
            
        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_list[ind1][ind2].append(bagging_intermediate[ind1][ind2])
    
    return SAA_list, bagging_list

        
def evaluation_final(SAA_list, bagging_list, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_list), len(bagging_list[0])
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_list[ind1][ind2][i][j])
    
    solution_obj_values = {str(solution): 0 for solution in all_solutions}

    for solution in all_solutions:
        obj_value = paretoObj(solution, sample_args['params'], *prob_args)
        solution_obj_values[str(solution)] = obj_value
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_obj_list, bagging_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list = []
                for j in range(number_of_iterations):
                    bagging_obj = solution_obj_values[str(bagging_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
    
    return SAA_obj_list, SAA_obj_avg, bagging_obj_list, bagging_obj_avg


def find_parameters(B_list, k_list, number_of_iterations,sample_number, rng):
    # currently only supports len(B_list) = 1 and len(k_list) = 2
    parameters= []
    results_SAA = []
    results_bagging = []
    for iter in range(40):
        
        cost = float(np.random.uniform(0.01, 0.1))
        price = float(cost + np.random.uniform(0.5, 5))
        params = float(np.random.uniform(1.0001, 1.1))
        
        sample_args = {
            'type': 'pareto',
            'params': params
        }

        tic = time.time()
        
        SAA_list, bagging_list = comparison_final(B_list, k_list, number_of_iterations, sample_number, rng, sample_args, price, cost)
        _, SAA_obj_avg, _, bagging_obj_avg = evaluation_final(SAA_list, bagging_list, sample_args, price, cost)
        
        print(f"Iteration {iter}, time: {time.time()-tic}")

        if all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][0])) and all(x >= y for x, y in zip(SAA_obj_avg, bagging_obj_avg[0][1])):
            continue
        else:
            name = [iter, [price, cost], params]
            plot_params(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list, name)
            parameters.append([price, cost, sample_args])
            results_SAA.append(SAA_obj_avg)
            results_bagging.append(bagging_obj_avg[0])
        
    return parameters, results_SAA, results_bagging

def plot_params(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list, name):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'B={B_list[ind1]}, k={k_list[ind2]}')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend()
    fig_name  = str(name[0]) + '_' + str(name[1]) + '_' + str(name[2]) + ".png"
    plt.savefig(fig_name)
    # plt.show()
    return


if __name__ == "__main__":
    rng = np.random.default_rng(seed=2024)
    B_list = [200]
    k_list = [0.1,50]
    number_of_iterations = 30
    sample_number = np.array([2**i for i in range(6, 14)])

    parameters, results_SAA, results_bagging = find_parameters(B_list, k_list, number_of_iterations, sample_number, rng)
    results = {'parameters': parameters, 'results_SAA': results_SAA, 'results_bagging': results_bagging}
    
    with open('results_newsvendor.json', 'w') as f: 
        json.dump(results, f)

