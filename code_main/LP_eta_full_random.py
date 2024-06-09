import time
import json
import numpy as np
from utils.LP_functions_full_random import LP_eta_comparison, get_pareto_params
import datetime

def serialize_data(data):
    if isinstance(data, dict):
        # Create a new dictionary that will hold serializable keys and values
        serialized_data = {}
        for key, value in data.items():
            # Convert tuple keys to a string representation
            if isinstance(key, tuple):
                key = str(key)
            # Recursively serialize the values
            serialized_data[key] = serialize_data(value)
        return serialized_data
    elif isinstance(data, list):
        # Apply serialization to each item in the list
        return [serialize_data(item) for item in data]
    elif isinstance(data, (int, float, str, bool)):
        # Basic types that are already JSON serializable
        return data
    elif isinstance(data, datetime.datetime):
        # Convert datetime to a string
        return data.isoformat()
    else:
        # Fallback for other types, convert them to string
        return str(data)

def is_json_serializable(data):
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False

def ensure_json_serializable(data):
    # First check if the data is already serializable
    if is_json_serializable(data):
        return data
    else:
        # If not, process the data to make it serializable
        return serialize_data(data)


if __name__ == "__main__":
    seed = 2024
    N = 8
    w =  {(1, 2): 2.5,
    (1, 3): 2.5,
    (1, 4): 2.5,
    (1, 5): 3,
    (1, 6): 3,
    (1, 7): 3,
    (1, 8): 3,
    (2, 3): 2.5,
    (2, 4): 2.5,
    (2, 5): 3,
    (2, 6): 3,
    (2, 7): 3,
    (2, 8): 3,
    (3, 4): 2.5,
    (3, 5): 3,
    (3, 6): 3,
    (3, 7): 3,
    (3, 8): 3,
    (4, 5): 3,
    (4, 6): 3,
    (4, 7): 3,
    (4, 8): 3,
    (5, 6): 3,
    (5, 7): 3,
    (5, 8): 3,
    (6, 7): 3,
    (6, 8): 3,
    (7, 8): 3}

    params = get_pareto_params(N,w)
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    sample_args = {
        "type" : "pareto",
        "params": params
    }

    rng_sample = np.random.default_rng(seed=seed)

    delta_list = [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6]
    epsilon = 2.5
    n = 50
    number_of_iterations = 500

    tic = time.time()
    SAA_eta_list, Alg34_eta_list, retrieved_set, x_count_dict, obj_opt_dict, pk_dict_SAA, pk_dict_Alg34 = LP_eta_comparison(delta_list, epsilon, n, number_of_iterations, rng_sample, sample_args, N, w, A)
    
    with open(str(n) + "solution_lists_eta.json", "w") as f:
        json.dump({"SAA_eta": SAA_eta_list, "Alg34_eta": Alg34_eta_list, "delta_list": delta_list, "epsilon": epsilon}, f)
    
    # with open("retrieved_set.json", "w") as f:
    #     json.dump({"retrieved_set": retrieved_set, "x_count_dict": x_count_dict, "obj_opt_dict": obj_opt_dict, "pk_dict_SAA": pk_dict_SAA, "pk_dict_Alg34": pk_dict_Alg34}, f)

    # save the remaining things just in a txt file, so that no need to worry about the format
    
    # with open("remaining.json", "w") as f:
    #     json.dump({"retrieved_set": serialize_data(retrieved_set), "x_count_dict": serialize_data(x_count_dict), "obj_opt_dict": serialize_data(obj_opt_dict), "pk_dict_SAA": serialize_data(pk_dict_SAA), "pk_dict_Alg34": serialize_data(pk_dict_Alg34)}, f)
