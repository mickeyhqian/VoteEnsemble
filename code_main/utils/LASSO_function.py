import numpy as np 
import time
from sklearn.linear_model import Lasso
from utils.generateSamples import genSample_LASSO
from ParallelSolve import majority_vote, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit, solve_model_selection


def LASSO_solve(sample_train, lambda_val):
    # use training dataset to solve the LASSO problem, get corresponding beta for a given lambda value
    # sample_k is a numpy 2-d array, the first column is y, the rest are x
    y = sample_train[:,0]
    X = sample_train[:,1:]
    lasso = Lasso(alpha=lambda_val, fit_intercept=False, random_state=2024, max_iter=50000)
    lasso.fit(X, y)
    return lasso.coef_ # which is a numpy array

def get_betas(sample_train, lambda_list):
    # get the beta values, which is essentially our decision space
    # run this script before the main algorithm
    # key is the lambda_val, value is the beta vector
    beta_dict = {}
    for lambda_val in lambda_list:
        tic = time.time()
        beta_dict[lambda_val] = LASSO_solve(sample_train, lambda_val)
        print(f"lambda {lambda_val} time: {time.time()-tic}")
    return beta_dict

def get_expected_loss(lambda_val, sample_args, beta_dict):
    # compute the expected MSE loss || y - X*beta ||^2, ignore the constant term arising from the noise
    beta = beta_dict[lambda_val]
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

def get_expected_loss_dict(sample_args, beta_dict):
    expected_loss_dict = {}
    for lambda_val in beta_dict.keys():
        loss = get_expected_loss(lambda_val, sample_args, beta_dict)
        expected_loss_dict[lambda_val] = loss
    return expected_loss_dict

def get_sample_loss(sample_k, beta):
    # evaluate the MSE loss of a given beta based on samples 
    y = sample_k[:,0]
    X = sample_k[:,1:]
    return np.mean((y - np.dot(X, beta))**2)

def loss_evaluation_wSol(sample_k, lambda_val, beta_dict):
    # evaluate the normalized loss of a certain lambda
    # return "- loss", to be used in Algorithms 3 and 4
    return - get_sample_loss(sample_k, beta_dict[lambda_val])/beta_dict[lambda_val].shape[0], lambda_val

def problem_initialization(lambda_list, sample_number_train, rng_sample_train, sample_args):
    # the seed rng_sample_train is only used for generating the training dataset here
    sample_train = genSample_LASSO(sample_number_train, rng_sample_train, sample_args)
    beta_dict = get_betas(sample_train, lambda_list)
    return beta_dict

def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number, rng_sample, rng_alg, sample_args, *prob_args):
    # note, prob_args for this problem is beta_dict
    SAA_list = []
    bagging_alg1_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg1_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_LASSO(n, rng_sample, sample_args)
            SAA, _ = majority_vote(sample_n, 1, n, solve_model_selection, rng_alg, *prob_args)
            SAA_intermediate.append(SAA)
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _  = majority_vote(sample_n, B, k, solve_model_selection, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(bagging)
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, solve_model_selection, loss_evaluation_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, solve_model_selection, loss_evaluation_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(bagging_alg3)
                    bagging_alg4_intermediate[ind1][ind2].append(bagging_alg4)
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")
            
        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_alg1_list[ind1][ind2].append(bagging_alg1_intermediate[ind1][ind2])
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list


def evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, expected_loss_dict):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_alg1_list), len(bagging_alg1_list[0])
    B12_list_len = len(bagging_alg3_list)
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg1_obj_list, bagging_alg1_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = expected_loss_dict[SAA_list[i][j]]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list = []
                for j in range(number_of_iterations):
                    bagging_obj = expected_loss_dict[bagging_alg1_list[ind1][ind2][i][j]]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_alg1_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_alg1_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
            
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = expected_loss_dict[bagging_alg3_list[ind1][ind2][i][j]]
                    bagging_obj_4 = expected_loss_dict[bagging_alg4_list[ind1][ind2][i][j]]
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg



