import numpy as np
import time
from utils.generateSamples import genSample_LR
from ParallelSolveNN import majority_vote_LP, baggingTwoPhase_woSplit_LP, baggingTwoPhase_wSplit_LP, train_network, sequentialEvaluate
from multiprocessing import Queue, Process

import torch
import torch.nn as nn
import numpy as np

class RegressionNN(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(RegressionNN, self).__init__()
        layers = []
        previous_size = input_size

        for size in layer_sizes:
            layers.append(nn.Linear(previous_size, size))
            layers.append(nn.ReLU())
            previous_size = size

        layers.append(nn.Linear(previous_size, 1))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def get_sample_based_loss_wSol(sample_k, model_params, layer_sizes):   
    # note this function can also be used for the evaluation of the model
    # during evalution, we use a large amount of noise free samples
    X = torch.Tensor(sample_k[:, 1:])
    y_true = torch.Tensor(sample_k[:, 0]).unsqueeze(1)
    input_size = X.shape[1]

    # deduce the parameter shapes from architecture
    def compute_shapes(input_size, layer_sizes):
        shapes = []
        previous_size = input_size
        for size in layer_sizes:
            weight_shape = (size, previous_size)
            bias_shape = (size,)
            shapes.append(weight_shape)
            shapes.append(bias_shape)
            previous_size = size
        # Add the output layer's weight and bias shapes
        shapes.append((1, previous_size))  # Output layer weight shape
        shapes.append((1,))  # Output layer bias shape
        return shapes

    # restore the model from the flat parameters
    def restore_model_from_flat(flat_params, input_size, layer_sizes):
        if isinstance(flat_params, tuple):
            flat_params = np.array(flat_params)

        model = RegressionNN(input_size, layer_sizes)
        shapes = compute_shapes(input_size, layer_sizes)
        idx = 0
        params_with_shapes = []
        for i, shape in enumerate(shapes):
            num_elements = np.prod(shape)
            param_reshaped = flat_params[idx:idx + num_elements].reshape(shape)
            params_with_shapes.append(param_reshaped)
            idx += num_elements

        with torch.no_grad():
            for i, (p, p_numpy) in enumerate(zip(model.parameters(), params_with_shapes)):
                p.copy_(torch.from_numpy(p_numpy))

        return model
    
    model = restore_model_from_flat(model_params, input_size, layer_sizes)
    model.eval()

    with torch.no_grad():
        y_pred = model(X)

    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    
    return -loss.item(), model_params # note that we already took the average during the computation of the MSE



def comparison_twoPhase(k_list, B12_list, epsilon, tolerance, number_of_iterations, sample_number, rng_sample, rng_alg, sample_args, *prob_args, k2_tuple = None):
    # note that only layer_sizes is passed in prob_args
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
            SAA, _ = majority_vote_LP(sample_n, 1, n, train_network, rng_alg, *prob_args)
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
                    bagging_alg3, _, _, eps_dyn = baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, train_network, get_sample_based_loss_wSol, rng_alg, *prob_args, k2 = k2)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, train_network, get_sample_based_loss_wSol, rng_alg, *prob_args, k2 = k2)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([float(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, k2 = {k2}, Bagging Alg 3 & 4time: {time.time()-tic}", f"adaptive epsilon: {eps_dyn}")

        SAA_list.append(SAA_intermediate)

        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg3_list, bagging_alg4_list

def evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    # parallel evaluation for NN
    solution_obj_values = {str(solution): [] for solution in all_solutions}
    numProcesses = 9
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for solution in all_solutions:
        for i in range(eval_time):
            ######## places that need to change for different problems ########
            sample_large = genSample_LR(large_number_sample, rng_sample, sample_args, add_noise = False)
            ########
            taskLists[processIndex].append((sample_large, solution))
            processIndex = (processIndex + 1) % numProcesses
    
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            ######## places that need to change for different problems ########
            processList.append(Process(target = sequentialEvaluate, args = (get_sample_based_loss_wSol, queue, taskLists[i], *prob_args), daemon=True))
            ########

    for process in processList:
        process.start()
    
    for _ in range(len(all_solutions) * eval_time):
        obj_value, solution = queue.get()
        if solution is not None:
            solution_obj_values[str(solution)].append(-obj_value) # note that we use the maximization form when computing the sample-based loss, so we need to negate it
    
    for process in processList:
        process.join() 
    
    solution_obj_values = {key: np.mean(value) for key, value in solution_obj_values.items()}

    return solution_obj_values

def evaluation_twoPhase(SAA_list, bagging_alg3_list, bagging_alg4_list, large_number_sample, eval_time, rng_sample, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    k_list_len = len(bagging_alg3_list[0])
    B12_list_len = len(bagging_alg3_list)

    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B12_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])
    
    solution_obj_values = evaluation_process(all_solutions, large_number_sample, eval_time, rng_sample, sample_args, *prob_args)

    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))
    
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = solution_obj_values[str(bagging_alg3_list[ind1][ind2][i][j])]
                    bagging_obj_4 = solution_obj_values[str(bagging_alg4_list[ind1][ind2][i][j])]
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))
    
    return SAA_obj_list, SAA_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg
    
