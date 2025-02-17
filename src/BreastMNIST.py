import os
import random
import sys
import argparse

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

import torchvision
from models import VisionTransformer
from dataPreparation import get_BreastMNIST, get_PneumoniaMNIST, get_ChestMNIST, get_DermaMNIST, get_TissueMNIST

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
# from datasets import load_dataset
from medmnist import BreastMNIST
import copy
from torch.utils.data import DataLoader, Dataset, Subset
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedYogi, FedAvg, FedAdam, FedTrimmedAvg, FedProx
from flwr.simulation import run_simulation
from flwr.common import Context
import logging
import SavingStrategy

logging.getLogger("flwr").setLevel(logging.INFO) 
project_path = sys.path[0]
data_path = project_path + "\data"
print("Project Path:", project_path)

random.seed(42)

# Add argument parser at the beginning of the script
parser = argparse.ArgumentParser(description='Federated Learning with multiple strategies')
parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam', 'fedtrimmedavg', 'fedprox'], default='fedavg',
                    help='Choose federated learning strategy (default: fedavg)')
parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

args = parser.parse_args()

# Load the BreastMNIST dataset

train_loader, val_loader, test_loader, num_classes = get_BreastMNIST()

num_train_instances = len(train_loader.dataset)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# print the dimension of images to verify all loaders have the same dimensions
def print_dim(loader, text):
  print('---------'+text+'---------')
  print(len(loader.dataset))
  for image, label in loader:
    print(image.shape)
    print(label.shape)
    break
  
# visualize image
# Visualize some examples

import torch

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")



# TODO evtl dynamisch machen (num_patches , size etc.)
image_size = 28
embed_dim=256
hidden_dim=embed_dim*3
num_heads=8
num_layers=6
patch_size=7
num_patches=16
num_channels=1
dropout=0.2

model = VisionTransformer(embed_dim=embed_dim,
                          hidden_dim=hidden_dim,
                          num_heads=num_heads,
                          num_layers=num_layers,
                          patch_size=patch_size,
                          num_channels=num_channels,
                          num_patches=num_patches,
                          num_classes=num_classes,
                          dropout=dropout)

# Transfer to GPU
model.to(device)
model_restore = None #'/content/model_20230712_211204_0'
if model_restore is not None and os.path.exists(model_restore):
  model.load_state_dict(torch.load(model_restore))
  model.restored = True

# setup the loss function
loss_fn = torch.nn.CrossEntropyLoss()
# setup the optimizer with the learning rate
model_optimizer = optim.Adam(model.parameters(), lr=3e-4)
# set a scheduler to decay the learning rate by 0.1 on the 100th 150th epochs
model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer,
                                            milestones=[100, 150], gamma=0.1)

# Function to evaluate model performance on the test set
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze(-1).long()).sum().item()

    accuracy = correct / total
    return accuracy


# Create a function to train locally on a client
def train_local_model(model, train_loader, loss_fn, optimizer, device, epochs):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        losses = [] 
        for imgs, labels in train_loader:
            labels = labels.to(torch.float)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, labels.squeeze(-1).long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_history.append(np.mean(losses))
    return loss_history

# epochs = 20

# loss_hist = train_local_model(model, train_loader, loss_fn, model_optimizer, device, epochs)

# plt.plot(loss_hist)
# plt.show()

# result = evaluate_model(model, test_loader, device)

# print(f"RESULT: {result}")


NUM_CLIENTS = 3

print("len of train_set::", num_train_instances)
print("len of val_set::", len(val_loader.dataset))
print("len of test_set::", len(test_loader.dataset))


# # Define the digit exclusions for each client
# digit_exclusions = {
#     0: [5],                   # Client 0 does not have digit 5
#     1: [4],                # Client 1 does not have digits 3, 4
#     2: [9],                   # Client 2 does not have digit 9
#     3: [8]             # Client 3 does not have digits 8, 7, 6                  # Client 4 does not have digit 2
# }

# Initialize lists to hold client indices
client_indices = {i: [] for i in range(NUM_CLIENTS)}

# Initialize a global index pool and digit-specific pools
global_index_pool = list(range(num_train_instances))
global_val_pool = list(range(len(val_loader.dataset)))

client_train_datasets = {}
client_val_datasets = {}
client_indices_mapping = {}  # To store the indices allocated to each client

# for client_idx in range(len(digit_exclusions)):
for client_idx in range(NUM_CLIENTS):
    # Filter the global index pool for the current client
    # available_indices = [idx for idx in global_index_pool if train_set[idx][1] not in digit_exclusions[client_idx]]
    available_indices = global_index_pool
    available_val_indices = global_val_pool

    amount_client_samples = len(available_indices) // (NUM_CLIENTS - client_idx)
    amount_client_val_samples = len(available_val_indices) // (NUM_CLIENTS - client_idx)


    # Sample 10,000 unique indices for the current client
    sampled_indices = random.sample(available_indices, amount_client_samples)
    sampled_val_indices = random.sample(available_val_indices, amount_client_val_samples)
    

    # Assign to the client dataset and store the indices
    client_train_datasets[client_idx] = Subset(train_loader.dataset, sampled_indices)
    client_val_datasets[client_idx] = Subset(val_loader.dataset, sampled_val_indices)
    client_indices_mapping[client_idx] = sampled_indices
    
    # Remove the sampled indices from the global pool
    global_index_pool = [idx for idx in global_index_pool if idx not in sampled_indices]
    global_val_pool = [idx for idx in global_val_pool if idx not in sampled_val_indices]

# Create DataLoaders for each client
client_loaders = {
    client_idx: DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for client_idx, dataset in client_train_datasets.items()
}

client_val_loader = {
    client_idx: DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for client_idx, dataset in client_val_datasets.items()
}

# Calculate and print the length of each client's dataset
client_data_sizes = [len(client_train_datasets[i]) for i in range(NUM_CLIENTS)]
print("Client data sizes:", client_data_sizes)



# # Display unique labels for each client
# for client_idx in range(len(client_datasets)):
#     labels = [train_set[idx][1] for idx in client_datasets[client_idx].indices]
#     unique_labels = set(labels)
#     print(f"Client {client_idx} has the following labels: {sorted(unique_labels)}")


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


# Federated Learning Setup
global_model = copy.deepcopy(model)  # Global model
client_models = [copy.deepcopy(model) for _ in range(NUM_CLIENTS)]  # Local models for each client

# Training loop for Federated Learning
num_rounds = 5  # Number of communication rounds
epochs_per_client = 10  # Number of local epochs per client


clients = []
# Define FlowerClient before the training section
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_local_model(self.model, self.train_loader, loss_fn, 
                         optim.Adam(self.model.parameters(), lr=3e-4), 
                         self.device, epochs=epochs_per_client)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = evaluate_model(self.model, self.val_loader, self.device)
        return float(accuracy), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

# Training section
if args.strategy == 'fedyogi':
    # Initialize FedYogi strategy
    strategy = SavingStrategy.SaveModelStrategy(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        eta=args.eta,
        eta_l=args.eta_l,
        beta_1=args.beta1,
        beta_2=args.beta2,
        tau=args.tau
    )
    
    # Create server configuration
    server_config = ServerConfig(num_rounds=num_rounds)
    
    def client_fn(context: Context) -> Client:
        client_model = copy.deepcopy(model)
        #train_loader = client_loaders[int(cid)]
        train_loader = client_loaders[int(context.node_config["partition-id"])]
        #print(context.state)
        #val_loader = DataLoader(client_datasets[context.node_config["partition-id"]], batch_size=16, shuffle=True)    
        client = FlowerClient(client_model, train_loader, val_loader, device).to_client() 
        clients.append(1)
        return client
    
    # Run simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_config,
        strategy=strategy
    )
elif args.strategy == 'fedadam':
    strategy = FedAdam(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        eta=0.01,                        # Server-side learning rate
        beta_1=0.9,                      # Momentum factor
        beta_2=0.99,                     # Second moment factor
        tau=0.001                        # Controls the importance of the proximal term
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    def client_fn(cid: str) -> Client:
        client_model = copy.deepcopy(model)
        train_loader = client_loaders[int(cid)]
        return FlowerClient(client_model, train_loader, val_loader, device)
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_config,
        strategy=strategy
    )

elif args.strategy == 'fedprox':
    strategy = FedProx(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        proximal_mu=0.01                 # Proximal term parameter (controls how far local models can deviate)
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    def client_fn(cid: str) -> Client:
        client_model = copy.deepcopy(model)
        train_loader = client_loaders[int(cid)]
        return FlowerClient(client_model, train_loader, val_loader, device)
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_config,
        strategy=strategy
    )

elif args.strategy == 'fedtrimmedavg':
    strategy = FedTrimmedAvg(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,     # Für deine 3 Clients
        min_available_clients=NUM_CLIENTS,  
        fraction_fit=1.0,                # Alle verfügbaren Clients nutzen
        fraction_evaluate=1.0,           # Alle Clients für Evaluation nutzen
        beta=0.1                         # 10% von beiden Enden trimmen
    )
    server_config = ServerConfig(num_rounds=num_rounds)
    def client_fn(cid: str) -> Client:
        client_model = copy.deepcopy(model)
        train_loader = client_loaders[int(cid)]
        return FlowerClient(client_model, train_loader, val_loader, device)
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_config,
        strategy=strategy
    )

else:  # fedavg
    for round_num in range(num_rounds):
        print(f'\nCommunication Round {round_num+1}/{num_rounds}')

        # Synchronize client models with the updated global model
        for client_idx in range(NUM_CLIENTS):
            client_models[client_idx].load_state_dict(global_model.state_dict())
        
        # Train each client's model locally
        for client_idx in range(NUM_CLIENTS):
            print(f'Client {client_idx+1} training...')
            train_local_model(client_models[client_idx], client_loaders[client_idx], loss_fn, 
                            optim.Adam(client_models[client_idx].parameters(), lr=3e-4), device, epochs=epochs_per_client)
            accuracy = evaluate_model(client_models[client_idx], val_loader, device)
            print("accuracy:", accuracy)
        
        # Aggregate client models to update global model
        federated_averaging(global_model, client_models, client_data_sizes)


print(f"clients: {clients}")
result = evaluate_model(global_model, val_loader, device)

save_checkpoint(global_model, model_optimizer, num_rounds, filename="final_checkpoint.pth")

result = evaluate_model(global_model, val_loader, device)
print(f"Final Evaluation RESULT: {result}")


# Step 1: Save the initial model state before training
initial_model = copy.deepcopy(model)
initial_weights = initial_model.state_dict()

# Step 2: Load the trained model from checkpoint
checkpoint_path = "final_checkpoint.pth"#
if not os.path.exists(checkpoint_path):
    print("Checkpoint file not found!")
else:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    trained_weights = model.state_dict()

    # Step 3: Compare Weights Layer by Layer
    print("\nComparing Initial and Trained Model Weights:\n")
    
    for name in initial_weights.keys():
        init_w = initial_weights[name]
        trained_w = trained_weights[name]

        # Compute the absolute difference
        diff = torch.abs(init_w - trained_w)
        max_diff = diff.max().item()

        print(f"Layer: {name}")
        print(f"Max Absolute Difference: {max_diff:.6f}")

        # Print a few values from each model for verification
        print(f"Initial Weights (First 5): {init_w.view(-1)[:5].tolist()}")
        print(f"Trained Weights (First 5): {trained_w.view(-1)[:5].tolist()}\n")