import os
import random
import sys
import argparse
import logging
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from src.model_utils import VisionTransformer, load_model_from_parameters, evaluate_model, train_local_model
from dataPreparation import get_BreastMNIST, get_PneumoniaMNIST, get_ChestMNIST, get_DermaMNIST, get_TissueMNIST

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.common import Metrics, Context

import logging
import SavingStrategy

def federated_learning_process(model=None, dataset:str = "BreastMNIST", strategy=None, clients:int = 3, rounds:int = 5, epochs:int = 10):
    """
    Federated Learning Process

    Args:
        model: VisionTransformer model
        dataset: str
        strategy: SavingStrategy
        clients: int
        rounds: int
        epochs: int
        
    Returns:
        trained_model: trained VisionTransformer model
    """

    NUM_CLIENTS = clients
    NUM_ROUNDS = rounds
    NUM_EPOCHS_PER_CLIENT = epochs

    global_model = model

    if strategy is None:
        strategy = SavingStrategy.SaveModelFedAvg(
            initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
            min_fit_clients=NUM_CLIENTS,     # Für deine 3 Clients
            min_available_clients=NUM_CLIENTS,  
            fraction_fit=1.0,                # Alle verfügbaren Clients nutzen
            fraction_evaluate=1.0,           # Alle Clients für Evaluation nutzen
            beta=0.1                         # 10% von beiden Enden trimmen
        )



    logging.getLogger("flwr").setLevel(logging.INFO) 
    project_path = sys.path[0]
    print("Project Path:", project_path)

    random.seed(42)

    # Add argument parser at the beginning of the script


    # Load the BreastMNIST dataset
    if dataset == "BreastMNIST":
        train_loader, val_loader, test_loader, num_classes = get_BreastMNIST()
    elif dataset == "PneumoniaMNIST":
        train_loader, val_loader, test_loader, num_classes = get_PneumoniaMNIST()
    elif dataset == "ChestMNIST":
        train_loader, val_loader, test_loader, num_classes = get_ChestMNIST()
    elif dataset == "DermaMNIST":
        train_loader, val_loader, test_loader, num_classes = get_DermaMNIST()
    elif dataset == "TissueMNIST":
        train_loader, val_loader, test_loader, num_classes = get_TissueMNIST()

    num_train_instances = len(train_loader.dataset)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

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

    if global_model is None:
        global_model = VisionTransformer(embed_dim=embed_dim,
                                hidden_dim=hidden_dim,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                patch_size=patch_size,
                                num_channels=num_channels,
                                num_patches=num_patches,
                                num_classes=num_classes,
                                dropout=dropout)

    # Transfer to GPU
    global_model.to(device)

    # setup the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # setup the optimizer with the learning rate
    model_optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # set a scheduler to decay the learning rate by 0.1 on the 100th 150th epochs
    model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer,
                                                milestones=[100, 150], gamma=0.1)

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




    # Federated Learning Setup
    global_model = copy.deepcopy(model)  # Global model

    # Define FlowerClient before the training section
    class FlowerClient(flwr.client.NumPyClient):
        def __init__(self, model, train_loader, val_loader, device):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.losses = []
            

        def get_parameters(self, config):
            return [val.cpu().numpy() for val in self.model.state_dict().values()]

        def set_parameters(self, parameters):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            loss_history = train_local_model(self.model, self.train_loader, loss_fn, 
                            optim.Adam(self.model.parameters(), lr=3e-4), 
                            self.device, epochs=NUM_EPOCHS_PER_CLIENT)
            self.losses.append(loss_history)
            return self.get_parameters(config={}), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            accuracy, loss = evaluate_model(self.model, self.val_loader, loss_fn, self.device)

            return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


    def client_fn(cid: str) -> Client:
            client_model = copy.deepcopy(model)
            train_loader = client_loaders[int(cid)]
            return FlowerClient(client_model, train_loader, val_loader, device).to_client()
    
    client =  ClientApp(client_fn=client_fn)
    
    # server_config = ServerConfig(num_rounds=NUM_ROUNDS)
    
    # history = flwr.simulation.start_simulation(
    #         client_fn=client_fn,
    #         num_clients=NUM_CLIENTS,
    #         config=server_config,
    #         strategy=strategy
    #     )
    def server_fn(context: Context)-> ServerAppComponents:

        config = ServerConfig(num_rounds=NUM_ROUNDS)

        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)

    
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS
    )


    # load the model from the latest communication round and return it
    trained_model = load_model_from_parameters(model) 
    result, _ = evaluate_model(trained_model, test_loader, loss_fn, device)
    print(f"Achieved accuracy: {result}")

    return trained_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with multiple strategies')
    parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam', 'fedtrimmedavg', 'fedprox'], default='fedavg',
                        help='Choose federated learning strategy (default: fedavg)')
    parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
    parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

    args = parser.parse_args()

    federated_learning_process()