import torch
import torch.nn as nn
import numpy as np

# Define the model class
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

# Function to flatten model parameters to a numpy array
# def model_to_numpy_flat(model):
#     params = [param.detach().cpu().numpy().flatten() for param in model.parameters()]
#     flat_array = np.concatenate(params)
#     return flat_array

def model_to_numpy_flat(model):
    params = []
    for param in model.parameters():
        flat_param = param.detach().cpu().numpy().flatten()
        params.append(flat_param)
        # print(f"Flattening param: {flat_param}")  # Debug print to check the values being flattened
    flat_array = np.concatenate(params)
    return flat_array

# Function to restore the model from a flat numpy array
# def restore_model_from_flat(flat_params, input_size, layer_sizes):
#     model = RegressionNN(input_size, layer_sizes)
#     shapes = compute_shapes(input_size, layer_sizes)
#     idx = 0
#     params_with_shapes = []
#     for shape in shapes:
#         num_elements = np.prod(shape)
#         params_with_shapes.append(flat_params[idx:idx + num_elements].reshape(shape))
#         idx += num_elements
#     with torch.no_grad():
#         for p, p_numpy in zip(model.parameters(), params_with_shapes):
#             p.copy_(torch.from_numpy(p_numpy))
#     return model

def restore_model_from_flat(flat_params, input_size, layer_sizes):
    model = RegressionNN(input_size, layer_sizes)
    shapes = compute_shapes(input_size, layer_sizes)
    idx = 0
    params_with_shapes = []
    for i, shape in enumerate(shapes):
        num_elements = np.prod(shape)
        param_reshaped = flat_params[idx:idx + num_elements].reshape(shape)
        params_with_shapes.append(param_reshaped)
        print(f"Restoring Layer {i}, shape: {shape}, flat segment: {flat_params[idx:idx + num_elements]}")  # Debug
        idx += num_elements

    with torch.no_grad():
        for i, (p, p_numpy) in enumerate(zip(model.parameters(), params_with_shapes)):
            p.copy_(torch.from_numpy(p_numpy))
            print(f"Restored Layer {i} param: {p}")  # Debug to see restored parameters

    return model

# Helper function to compute shapes based on layer sizes
# def compute_shapes(input_size, layer_sizes):
#     shapes = []
#     previous_size = input_size
#     for size in layer_sizes:
#         weight_shape = (size, previous_size)
#         bias_shape = (size,)
#         shapes.append(weight_shape)
#         shapes.append(bias_shape)
#         previous_size = size
#     return shapes

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

# Function to check parameter equivalence
def check_parameters_numpy(model1, model2):
    params1 = model_to_numpy_flat(model1)
    params2 = model_to_numpy_flat(model2)
    print(params1, params2)
    return np.allclose(params1, params2, atol=1e-5)


def print_model_parameters(model):
    for idx, param in enumerate(model.parameters()):
        print(f"Layer {idx} parameters: {param.data}")


def check_parameters_detail(model1, model2):
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    if len(params1) != len(params2):
        print("Different number of parameter groups")
        return False

    all_close = True
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1, p2, atol=1e-5):
            print("Mismatch found:")
            print("Original:", p1)
            print("Restored:", p2)
            all_close = False
    return all_close

# Test the encode/decode functionality
input_size = 10
layer_sizes = [20, 30, 10, 50]
model = RegressionNN(input_size, layer_sizes)

# Initialize model with random weights
for param in model.parameters():
    param.data.uniform_(-1, 1)

# Encode model parameters to a flat numpy array
flat_params = model_to_numpy_flat(model)

# Decode the flat numpy array back into a new model
restored_model = restore_model_from_flat(flat_params, input_size, layer_sizes)

print("Original model parameters:")
print_model_parameters(model)

# print("Restored model parameters:")
# print_model_parameters(restored_model)

# Check parameters and print detailed mismatches
are_identical = check_parameters_detail(model, restored_model)
print("Parameters identical:", are_identical)