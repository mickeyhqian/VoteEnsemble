import numpy as np 
import time
from utils.generateSamples import genSample_LR
from ParallelSolve import majority_vote_LP, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, solve_LR

def get_expected_loss(beta, sample_args):
    # compute the expected MSE loss || y - X*beta ||^2, ignore the constant term arising from the noise
    meanX = sample_args["meanX"]
    shapes = sample_args["params"]
    beta_true = sample_args["beta_true"]
    # create a matrix for storing the second moment of the x vector, compute mean * mean^T
    M_x = np.outer(meanX, meanX)
    # compute the variance vector
    vars = [shape/((shape-1)**2 * (shape-2)) for shape in shapes]
    M_x += np.diag(vars)
    beta = np.array(beta) # in case beta is not a numpy array
    return np.dot(np.dot(beta-beta_true, M_x), beta-beta_true)

def get_sample_based_loss_wSol(sample_k, beta):
    y = sample_k[:,0]
    X = sample_k[:,1:]
    # intentionally return the negative cost, to be used in Algorithms 3 and 4 
    return - np.mean((y - np.dot(X, beta))**2), beta

def comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, k2_tuple = None):
    SAA_list = []
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_LR(n, rng_sample, sample_args)
            SAA, _ = majority_vote_LP(sample_n, 1, n, solve_LR, rng_alg)
            SAA_intermediate.append(tuple([float(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                if k2_tuple is not None:
                    k2 = max(k2_tuple[0], int(n*k2_tuple[1]))
                else:
                    k2 = None
                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, solve_LR, get_sample_based_loss_wSol, rng_alg, k2 = k2)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, solve_LR, get_sample_based_loss_wSol, rng_alg, k2 = k2)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, k2 = {k2}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
            
        SAA_list.append(SAA_intermediate)

        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg3_list, bagging_alg4_list

def evaluation_twoPhase(SAA_list, bagging_alg3_list, bagging_alg4_list, sample_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    k_list_len = len(bagging_alg3_list[0])
    B12_list_len = len(bagging_alg3_list)

    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]

    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = get_expected_loss(SAA_list[i][j], sample_args)
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = get_expected_loss(bagging_alg3_list[ind1][ind2][i][j], sample_args)
                    bagging_obj_4 = get_expected_loss(bagging_alg4_list[ind1][ind2][i][j], sample_args)
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg
