import time
import json
import numpy as np
from utils.portfolio_functions_continuous import portfolio_eta_comparison, get_pareto_params
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

    mu = np.array([
    1.9367129768889861,
    1.7033293134683836,
    1.2301708031827778,
    1.5004752383434894,
    1.6211307309110454,
    1.6204079110596386
  ])
    b = 1.5140166581891492

    params = get_pareto_params(mu)
    sample_args = {
        "type" : "pareto",
        "params": params
    }

    rng_sample = np.random.default_rng(seed=seed)

    delta_list = [2**i for i in range(-9,2)]
    epsilon = 0.02
    n = 50000
    B = 500
    k = 50
    # number_of_iterations = 1000
    number_of_iterations = 5000

    tic = time.time()
    SAA_eta_list, Alg34_eta_list, retrieved_set, x_count_dict, obj_opt_dict, SAA_conditional_pk, pk_dict_Alg34 = portfolio_eta_comparison(delta_list, B, k, epsilon, n, number_of_iterations, rng_sample, sample_args, mu, b)

    with open("solution_lists_eta.json", "w") as f:
        json.dump({"SAA_eta": SAA_eta_list, "Alg34_eta": Alg34_eta_list}, f)

    with open("remaining.json", "w") as f:
        json.dump({"retrieved_set": serialize_data(retrieved_set), "x_count_dict": serialize_data(x_count_dict), "obj_opt_dict": serialize_data(obj_opt_dict), "SAA_conditional_pk": serialize_data(SAA_conditional_pk), "pk_dict_Alg34": serialize_data(pk_dict_Alg34)}, f)
    
    parameters = {
        "seed": seed,
        "mu": mu.tolist(),
        "b": b,
        "sample_args": sample_args,
        "delta_list": delta_list,
        "epsilon": epsilon,
        "n": n,
        "B": B,
        "k": k,
        "number_of_iterations": number_of_iterations
    }

    with open("parameters.json", "w") as f:
        json.dump(parameters, f, indent = 2)

    