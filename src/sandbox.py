import torch



# Function for FedAvg (Federated Averaging)
def federated_averaging(global_model, client_models, client_data_sizes):
    global_dict = global_model.state_dict()
    total_data_points = sum(client_data_sizes)  # Total number of data points across all clients
    
    # Initialize the global state with zeros
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
    
    # Perform weighted averaging
    for i, client_model in enumerate(client_models):
        client_dict = client_model.state_dict()
        weight = client_data_sizes[i] / total_data_points  # Compute weight for this client
        for key in global_dict.keys():
            global_dict[key] += client_dict[key] * weight
    
    # Load the weighted average back into the global model
    global_model.load_state_dict(global_dict)




# # Step 1: Save the initial model state before training
#     initial_model = copy.deepcopy(model)
#     initial_weights = initial_model.state_dict()

#     # Step 2: Load the trained model from checkpoint
#     checkpoint_path = "final_checkpoint.pth"#
#     if not os.path.exists(checkpoint_path):
#         print("Checkpoint file not found!")
#     else:
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         trained_weights = model.state_dict()

#         # Step 3: Compare Weights Layer by Layer
#         print("\nComparing Initial and Trained Model Weights:\n")
        
#         for name in initial_weights.keys():
#             init_w = initial_weights[name]
#             trained_w = trained_weights[name]

#             # Compute the absolute difference
#             diff = torch.abs(init_w - trained_w)
#             max_diff = diff.max().item()

#             print(f"Layer: {name}")
#             print(f"Max Absolute Difference: {max_diff:.6f}")

#             # Print a few values from each model for verification
#             print(f"Initial Weights (First 5): {init_w.view(-1)[:5].tolist()}")
#             print(f"Trained Weights (First 5): {trained_w.view(-1)[:5].tolist()}\n")