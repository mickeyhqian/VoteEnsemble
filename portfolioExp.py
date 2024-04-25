import numpy as np
from SubprocessParallel import majority_vote
import pickle
import os


# define a generator for random samples
def sample_func(rng, distribution_name, **dist_params):
      # generate symmetric pareto distribution with mean = a
     if distribution_name == "sym_pareto":
         a, size = dist_params['a'], dist_params['size']
         samples = rng.pareto(2.1, size) * (np.array(rng.uniform(size=size) < 0.5, dtype=int) * 2 - 1)
         return samples + a    
     
     if hasattr(rng, distribution_name):
        dist_func = getattr(rng, distribution_name)
        samples = dist_func(**dist_params)
        return samples + 1 if distribution_name == "pareto" else samples
     else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")


def generate_covariance_matrix(m):
    A = np.random.rand(m, m)  
    cov_matrix = np.dot(A, A.T) 
    return cov_matrix


# function that evaluates the objective (no need to do a second stage gurobi)
def calculate_cvar(sample_xi, p, x, alpha=0.95):
    losses = - np.dot(sample_xi, np.array(x) * np.array(p))
    # var_threshold = np.quantile(losses, alpha)
    # cvar = np.mean(losses[losses > var_threshold])
    var_threshold = sorted(losses)[min(int(len(losses) * alpha), len(losses) - 1)]
    cvar = np.mean(np.maximum(0.0, losses - var_threshold)) + var_threshold
    
    return cvar


# SAA and majority vote comparison
def comparison(mu, dist_paras,p, b,B,number_of_iterations,ratio,sample_number, rng):
    name, paras = dist_paras["name"], dist_paras["paras"]
    
    SAA_list = []
    majority_list = []
    for n in sample_number:
        SAA_intermediate = []
        majority_intermediate = []
        for _ in range(number_of_iterations):
            if name == "multivariate_normal":
                sample_n = sample_func(rng, name, size=n, mean=mu, cov=paras)
            elif name == "sym_pareto":
                arrays_list = []
                for i in range(len(mu)):
                    arrays_list.append(sample_func(rng, name, size=n, a=paras[i]))
                sample_n = np.vstack(arrays_list).T
            else: 
                raise ValueError(f"Unsupported distribution: {name}")
            SAA = majority_vote(sample_n, 1, n, "portfolio", rng, p, mu, b)
            SAA_intermediate.append(SAA)
            majority = majority_vote(sample_n, B, int(n*ratio), "portfolio", rng, p, mu, b)
            majority_intermediate.append(majority)
            
        SAA_list.append(SAA_intermediate)
        majority_list.append(majority_intermediate)
    return SAA_list, majority_list

def evaluation(SAA_list, majority_list, mu, dist_paras, p, number_of_iterations, sample_number, large_number_sample, rng):
    name, paras = dist_paras["name"], dist_paras["paras"]
    if name == "multivariate_normal":
        large_sample = sample_func(rng, name, size=large_number_sample, mean=mu, cov=paras)
    elif name == "sym_pareto":
        arrays_list = []
        for i in range(len(mu)):
            arrays_list.append(sample_func(rng, name, size=large_number_sample, a=paras[i]))
        large_sample = np.vstack(arrays_list).T
    else:
        raise ValueError(f"Unsupported distribution: {name}")

    SAA_obj_list = []
    majority_obj_list = []
    for i in range(len(sample_number)):
        SAA_obj_list.append([])
        majority_obj_list.append([])
        for j in range(number_of_iterations):
            SAA_obj_list[-1].append(calculate_cvar(large_sample, p, SAA_list[i][j]))
            majority_obj_list[-1].append(calculate_cvar(large_sample, p, majority_list[i][j]))
    return SAA_obj_list, majority_obj_list


# script for finding repeatedly run experiments to find good parameters (pareto case)
def find_parameters(m,B,number_of_iterations,ratio,sample_number,large_number_sample):
    rng = np.random.default_rng(seed=888)
    # top_parameters = []
    for i in range(30):
        mu = rng.uniform(2, 5, m)
        dist_paras_pareto = {"name": "sym_pareto", "paras": mu}
        p = rng.uniform(0, 1, m)
        b = rng.uniform(1, 4)

        rngAlgo = np.random.default_rng(seed=2024)
        SAA_list, majority_list = comparison(mu, dist_paras_pareto, p, b, B, number_of_iterations, ratio, sample_number, rngAlgo)
        SAA_obj_list, majority_obj_list = evaluation(SAA_list, majority_list, mu, dist_paras_pareto, p, number_of_iterations, sample_number, large_number_sample, rngAlgo)

        # prop = sum((majority_obj_list[i] - SAA_obj_list[i])/majority_obj_list[i] for i in range(len(sample_number)//2,len(sample_number)))
        # top_parameters.append((prop, [mu.tolist(), p.tolist(), b], SAA_list, majority_list, SAA_obj_list, majority_obj_list))
        # top_parameters.sort(key = lambda x: x[0], reverse = True)


        resultDir = "/home/hqian/ResearchProjects/result/portfolio"
        os.makedirs(resultDir, exist_ok=True)
        with open(os.path.join(resultDir, f"param_{dist_paras_pareto['name']}_{i}"), "wb") as f:
            pickle.dump(([mu.tolist(), p.tolist(), b], SAA_list, majority_list, SAA_obj_list, majority_obj_list), f)
    
    # results = []
    # for prop, params, SAA_list, majority_list, SAA_obj_list, majority_obj_list in top_parameters:
    #     result = {
    #         "parameters": params,
    #         'SAA_list': SAA_list,
    #         'majority_list': majority_list,
    #         'SAA_obj_list': SAA_obj_list,
    #         'majority_obj_list': majority_obj_list
    #     }
    #     results.append(result)
    
    # return results
            

if __name__ == "__main__":
    m = 6
    B = 400
    number_of_iterations = 20

    # B = 10
    # number_of_iterations = 2

    ratio = 0.25
    sample_number = np.array([2**i for i in range(5, 16)])
    # sample_number = np.array([2**i for i in range(5, 6)])
    large_number_sample = 1000000
    results = find_parameters(m, B, number_of_iterations, ratio, sample_number, large_number_sample)
