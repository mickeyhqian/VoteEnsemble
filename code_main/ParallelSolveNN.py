# NN version
import numpy as np
from multiprocessing import Queue, Process
import time

############# Utility functions for parallel computation #############
def sequentialSolve(opt_func, queue: Queue, sampleList, *prob_args):
    for sample in sampleList:
        queue.put(opt_func(sample, *prob_args))

def sequentialEvaluate(eval_func, queue: Queue, taskList, *prob_args):
    # this evaluation function is used for final solution evaluation
    # taskList: a list of tuples, tuple[0] is the sample and tuple[1] is the solution
    for task in taskList:
        queue.put(eval_func(task[0], task[1], *prob_args))

def sequentialEvaluate_twoPhase(eval_func, queue: Queue, taskList, *prob_args):
    # this evaluation function is used for the second stage of bagging
    for task in taskList:
        queue.put(solution_evaluation(task[0], task[1], eval_func, *prob_args))

############# Algorithm 1 #############
def majority_vote(sample_n, B, k, opt_func, rng, *prob_args,varepsilon=None):
    x_count = {}
    numProcesses = 9
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[rng.choice(sample_n.shape[0], k, replace=False)])
        processIndex = (processIndex + 1) % numProcesses

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            args = (opt_func, queue, sampleLists[i]) + prob_args
            if varepsilon is not None:
                args += (varepsilon,)
            processList.append(Process(target=sequentialSolve, args=args, daemon=True))
            # processList.append(Process(target=sequentialSolve, args=(opt_func, queue, sampleLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()

    for _ in range(B):
        x_k = queue.get()
        if x_k is not None:
            sol = x_k if type(x_k) == int or type(x_k) == float else tuple(int(entry) for entry in x_k)
            x_count[sol] = x_count.get(sol, 0) + 1

    for process in processList:
        process.join()
    
    x_max = max(x_count, key=x_count.get)
    return x_max, list(x_count.keys())


############# Algorithms 3 and 4 #############
def solution_evaluation(sample_k, retrieved_solutions, eval_func, *prob_args):
    # output: a numpy array of evaluations of the retrieved solutions
    # assume the problem is a maximization problem
    evaluations = []
    max_obj = -np.inf
    for sol in retrieved_solutions:
        obj = eval_func(sample_k, sol, *prob_args)
        obj = obj[0] if len(obj) >= 2 else obj
        max_obj = max(max_obj, obj)
        evaluations.append(obj)
    return np.array(evaluations), max_obj

def get_adaptive_epsilon(suboptimality_gap_matrix, tolerance):
    # implement the strategy of dynamic epsilon, use the bisection method to find the proper epsilon value
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, 0)
    if np.max(x_ratio) >= 0.5:
        return 0
    
    left, right = 0, 1
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, right)
    while np.max(x_ratio) < 0.5:
        left = right
        right *= 2
        x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, right)
    
    while right - left > tolerance:
        mid = (left + right) / 2
        x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, mid)
        if np.max(x_ratio) >= 0.5:
            right = mid
        else:
            left = mid
    
    return right

def get_suboptimality_gap(evaluation_matrix, max_obj_vector):
    # calculate the suboptimal gap, assume it is a maximization problem
    # simply use the max_obj_vector to minus the evaluation_matrix
    max_obj_vector = max_obj_vector.reshape(-1,1)
    return max_obj_vector - evaluation_matrix

def get_epsilon_optimal(suboptimality_gap_matrix, epsilon):
    # for the given suboptimal gap matrix, return the ratio of within epsilon-optimal
    # note that suboptimality_gap is calculated as max_obj - obj
    return np.sum(suboptimality_gap_matrix <= epsilon, axis=0)/suboptimality_gap_matrix.shape[0]

def get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon):
    # get the phase II solution based on the suboptimal gap matrix
    x_ratio = get_epsilon_optimal(suboptimality_gap_matrix, epsilon)
    ind_max = np.argmax(x_ratio)
    return retrieved_solutions[ind_max]

def get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k, eval_func, rng, *prob_args):
    # generate a evaluation matrix, where i,j-th entry is the evaluation of the j-th solution in the i-th resample
    # i.e., each resample get a new row of evaluations
    # then, for each solution, calculate its suboptimal gap in each resample
    tic = time.time()
    evaluation_matrix = np.zeros((B2, len(retrieved_solutions)))
    max_obj_vector = np.zeros(B2)

    numProcesses = 9
    taskLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B2):
        sample_k = sample_n[rng.choice(sample_n.shape[0], k, replace=False)]
        taskLists[processIndex].append((sample_k, retrieved_solutions))
        processIndex = (processIndex + 1) % numProcesses
    
    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(taskLists[i]) > 0:
            processList.append(Process(target=sequentialEvaluate_twoPhase, args = (eval_func, queue, taskLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()
    
    for iter in range(B2):
        evaluations_array, max_obj = queue.get()
        evaluation_matrix[iter] = evaluations_array
        max_obj_vector[iter] = max_obj

    for process in processList:
        process.join()

    suboptimality_gap_matrix = get_suboptimality_gap(evaluation_matrix, max_obj_vector)
    print(f"Time for generating suboptimality gap matrix: {time.time()-tic}")
    return suboptimality_gap_matrix

# Note: only Phase I is related to specific algorithms, e.g., SAA, DRO. Phase II is a general solution ranking stage.
# Note: retrieve_solutions is a list of solutions, so it is ordered
def baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, varepsilon = None):
    # implementation of Algorithm 3
    # epsilon: can either be a numerical value or "dynamic", which stands for using the bisection method to find a proper value
    _, retrieved_solutions = majority_vote(sample_n, B1, k, opt_func, rng, *prob_args, varepsilon=varepsilon)

    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    suboptimality_gap_matrix = get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, 0)
        epsilon = 0
    else: # epsilon == "dynamic" and more than two solutions
        epsilon = get_adaptive_epsilon(suboptimality_gap_matrix, tolerance)
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
        
    return x_max, suboptimality_gap_matrix, retrieved_solutions, epsilon
    

def baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, varepsilon = None):
    # implementation of Algorithm 4
    sample_n1 = sample_n[:int(sample_n.shape[0]/2)]
    sample_n2 = sample_n[int(sample_n.shape[0]/2):]

    _, retrieved_solutions = majority_vote(sample_n1, B1, k, opt_func, rng, *prob_args, varepsilon=varepsilon)
    
    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    suboptimality_gap_matrix_n2 = get_suboptimality_gap_matrix(sample_n2, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    if len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, 0)
        epsilon = 0
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    # epsilon == "dynamic" and more than two solutions
    suboptimality_gap_matrix_n1 = get_suboptimality_gap_matrix(sample_n1, retrieved_solutions, B2, k, eval_func, rng, *prob_args)
    epsilon = get_adaptive_epsilon(suboptimality_gap_matrix_n1, tolerance)
    x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
    return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon


############# LP version Algorithms #############
def check_exist(x_count, x):
    if x_count == {}:
        return None
    eps = 1e-8
    for key in x_count:
        if np.linalg.norm(np.array(key) - np.array(x)) < eps:
            return key
    return None
        

def majority_vote_LP(sample_n, B, k, opt_func, rng, *prob_args):
    x_count = {}
    numProcesses = 9
    sampleLists = [[] for _ in range(numProcesses)]
    processIndex = 0
    for _ in range(B):
        # choose k samples from total n samples
        sampleLists[processIndex].append(sample_n[rng.choice(sample_n.shape[0], k, replace=False)])
        processIndex = (processIndex + 1) % numProcesses

    processList: list[Process] = []
    queue = Queue()
    for i in range(numProcesses):
        if len(sampleLists[i]) > 0:
            processList.append(Process(target=sequentialSolve, args=(opt_func, queue, sampleLists[i], *prob_args), daemon = True))
    
    for process in processList:
        process.start()

    for _ in range(B):
        x_k = queue.get()
        if x_k is not None:
            sol = x_k if type(x_k) == int or type(x_k) == float else tuple(entry for entry in x_k)
            key = check_exist(x_count, sol)
            if key is not None:
                x_count[key] += 1
            else:
                x_count[sol] = 1
    
    for process in processList:
        process.join()

    x_max = max(x_count, key=x_count.get)
    return x_max, list(x_count.keys())


# Note, functions baggingPhaseII and baggingPhaseII_epsilonDynamic need not to be modified for the LP version
def baggingTwoPhase_woSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, k2 = None):
    _, retrieved_solutions = majority_vote_LP(sample_n, B1, k, opt_func, rng, *prob_args)
    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    if k2 is None:
        k2 = k

    suboptimality_gap_matrix = get_suboptimality_gap_matrix(sample_n, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    elif len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, 0)
        epsilon = 0
    else: # epsilon == "dynamic" and more than two solutions
        epsilon = get_adaptive_epsilon(suboptimality_gap_matrix, tolerance)
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix, epsilon)
    
    return x_max, suboptimality_gap_matrix, retrieved_solutions, epsilon


def baggingTwoPhase_wSplit_LP(sample_n, B1, B2, k, epsilon, tolerance, opt_func, eval_func, rng, *prob_args, k2 = None):
    # implementation of Algorithm 4
    sample_n1 = sample_n[:int(sample_n.shape[0]/2)]
    sample_n2 = sample_n[int(sample_n.shape[0]/2):]

    _, retrieved_solutions = majority_vote_LP(sample_n1, B1, k, opt_func, rng, *prob_args)

    if len(retrieved_solutions) == 1:
        return retrieved_solutions[0], {retrieved_solutions[0]: B2}, retrieved_solutions, 0
    
    if k2 is None:
        k2 = k

    suboptimality_gap_matrix_n2 = get_suboptimality_gap_matrix(sample_n2, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    if type(epsilon) is not str:
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    if len(retrieved_solutions) == 2: # epsilon == "dynamic" and only two solutions
        x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, 0)
        epsilon = 0
        return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon
    
    # epsilon == "dynamic" and more than two solutions
    suboptimality_gap_matrix_n1 = get_suboptimality_gap_matrix(sample_n1, retrieved_solutions, B2, k2, eval_func, rng, *prob_args)
    epsilon = get_adaptive_epsilon(suboptimality_gap_matrix_n1, tolerance)
    x_max = get_PhaseII_solution(retrieved_solutions, suboptimality_gap_matrix_n2, epsilon)
    return x_max, suboptimality_gap_matrix_n2, retrieved_solutions, epsilon


############# Algorithms for testing #############
def test_baggingTwoPhase_woSplit_LP(sample_n, B1, k, opt_func, eval_func, rng, *prob_args):
    # assume the problem is a minimization problem
    # here the eval_func is the exact evaluation function
    # retrieved_solutions is a list of tuples
    _, retrieved_solutions = majority_vote_LP(sample_n, B1, k, opt_func, rng, *prob_args)
    # compute the average of all solution vectors
    avg_sol = np.mean(retrieved_solutions, axis=0)
    avg_sol = tuple(float(entry) for entry in avg_sol)
    obj_avg_sol = eval_func(avg_sol, *prob_args)
        
    obj_dict = {sol: eval_func(sol, *prob_args) for sol in retrieved_solutions}
    # return the solution with the minimum objective value
    x_min = min(obj_dict, key=obj_dict.get)
    obj_min = obj_dict[x_min]
    
    x_min_avg = avg_sol if obj_avg_sol < obj_min else x_min
    obj_min_avg = obj_avg_sol if obj_avg_sol < obj_min else obj_min

    return x_min, obj_min, x_min_avg, obj_min_avg




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

############# Neural network #############
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
    
def prepare_data(X, y, batch_size=16):
    tensor_x = torch.Tensor(X)  # Convert features to Torch tensor
    tensor_y = torch.Tensor(y).unsqueeze(1)  # Convert labels to Torch tensor and add an extra dimension

    dataset = TensorDataset(tensor_x, tensor_y)  # Create dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader
    return dataloader

# convert the model parameters to a 1-d numpy array (flat)
def model_to_numpy_flat(model):
    params = []
    for param in model.parameters():
        flat_param = param.detach().cpu().numpy().flatten()
        params.append(flat_param)
    flat_array = np.concatenate(params)
    return flat_array

def train_network(sample_n, layer_sizes, epochs=100, learning_rate=1e-3):
    # the only required parameter is layer_sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = sample_n[:, 1:]  # Extract features
    y = sample_n[:, 0]   # Extract labels

    dataloader = prepare_data(X, y)
    input_size = X.shape[1]
    model = RegressionNN(input_size, layer_sizes).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    params = model_to_numpy_flat(model) # which is a list of numpy matrices/arrays
    return params # return the numpy parameters of the trained model (to be used in bagging algorithms)







