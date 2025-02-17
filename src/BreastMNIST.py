import os
import random
import sys
import argparse

import torch
import torch.optim as optim

import torchvision
from models import VisionTransformer

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
from flwr.server.strategy import FedYogi
from flwr.simulation import run_simulation
from flwr.common import Context
import logging

logging.getLogger("flwr").setLevel(logging.ERROR)  # Suppress warnings, only show errors
project_path = sys.path[0]
data_path = project_path + "\data"

random.seed(42)

# Add argument parser at the beginning of the script
parser = argparse.ArgumentParser(description='Federated Learning with FedAvg, FedAdam or FedYogi')
parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam'], default='fedavg',
                    help='Choose federated learning strategy (default: fedavg)')
parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

args = parser.parse_args()

# get dataset
MNIST_preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5,), (0.5,))])
if not os.path.exists(data_path):
    os.makedirs(data_path)
train_breast = BreastMNIST(split="train", download=True, root=project_path + "\data", transform=MNIST_preprocess)
val_breast = BreastMNIST(split="val", download=True, root=project_path + "\data", transform=MNIST_preprocess)
test_breast = BreastMNIST(split="test", download=True, root=project_path + "\data", transform=MNIST_preprocess)

train_loader = DataLoader(train_breast, batch_size=16, shuffle=True)
val_loader = DataLoader(val_breast, batch_size=16, shuffle=True)
test_loader = DataLoader(test_breast, batch_size=16, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Define FlowerClient before the training section
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
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
        accuracy = evaluate_model(self.model, self.test_loader, self.device)
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

# Define client function with Context
def client_fn(context: Context) -> Client:
    cid = context.cid
    client_model = copy.deepcopy(model)
    train_loader = client_loaders[int(cid)]
    numpy_client = FlowerClient(client_model, train_loader, test_loader, device)
    return numpy_client.to_client()

# Training section
if args.strategy == 'fedyogi':
    # Initialize FedYogi strategy
    strategy = FedYogi(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        eta=args.eta,
        eta_l=args.eta_l,
        beta_1=args.beta1,
        beta_2=args.beta2,
        tau=args.tau
    )
    
    # Create server configuration
    server_config = ServerConfig(num_rounds=num_rounds)
    
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
        eta=args.eta,
        eta_l=args.eta_l,
        beta_1=args.beta1,
        beta_2=args.beta2,
        tau=args.tau
    )
    server_config = ServerConfig(num_rounds=num_rounds)
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
            accuracy = evaluate_model(client_models[client_idx], test_loader, device)
            print("accuracy:", accuracy)
        
        # Aggregate client models to update global model
        federated_averaging(global_model, client_models, client_data_sizes)

    result = evaluate_model(global_model, test_loader, device)
    print(f"RESULT: {result}")