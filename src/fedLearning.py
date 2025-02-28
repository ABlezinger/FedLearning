import random
import sys
import argparse
import logging
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import flwr
from flwr.client import Client, ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.common import Context

from model_utils import VisionTransformer, load_model_from_parameters, evaluate_model, train_local_model
from dataPreparation import get_BreastMNIST, get_PneumoniaMNIST, get_ChestMNIST

import SavingStrategy

def federated_learning_process(model=None, dataset:str = "BreastMNIST", strategy=None, clients:int = 3, rounds:int = 5, epochs:int = 10):
    """
    Federated Learning Process
    
    This function implements a complete federated learning workflow using Flower framework.
    It distributes the training process across multiple clients, aggregates their model updates,
    and produces a global model through collaborative training.
    
    Args:
        model: VisionTransformer model - The initial model architecture to be trained
        dataset: str - Name of the dataset to use ("BreastMNIST", "PneumoniaMNIST", or "ChestMNIST")
        strategy: SavingStrategy - The federated learning strategy to use for aggregation
        clients: int - Number of client nodes to simulate in the federated setup
        rounds: int - Number of communication rounds between server and clients
        epochs: int - Number of local training epochs per client per round
        
    Returns:
        trained_model: VisionTransformer - The trained global model after federated learning
        results: dict - Dictionary containing the test results (accuracy, f1)
    """

    NUM_CLIENTS = clients
    NUM_ROUNDS = rounds
    NUM_EPOCHS_PER_CLIENT = epochs

    global_model = model

    # Set up default strategy if none provided (FedAvg with trimming)
    if strategy is None:
        strategy = SavingStrategy.SaveModelFedAvg(
            initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
            min_fit_clients=NUM_CLIENTS,     # Ensure all clients participate
            min_available_clients=NUM_CLIENTS,  
            fraction_fit=1.0,                # Use all available clients
            fraction_evaluate=1.0,           # Use all clients for evaluation
            beta=0.1                         # Trim 10% from both ends (for robustness)
        )

    # Configure logging for Flower framework
    logging.getLogger("flwr").setLevel(logging.INFO) 
    project_path = sys.path[0]
    print("Project Path:", project_path)

    # Set random seed for reproducibility
    random.seed(42)

    # Load the appropriate medical MNIST dataset based on selection
    if dataset == "BreastMNIST":
        train_loader, val_loader, _ , _ = get_BreastMNIST()
    elif dataset == "PneumoniaMNIST":
        train_loader, val_loader, _, _ = get_PneumoniaMNIST()
    elif dataset == "ChestMNIST":
        train_loader, val_loader, _, _ = get_ChestMNIST()

    num_train_instances = len(train_loader.dataset)

    # Determine the computation device (MPS for Mac M-series, fallback to CPU)
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    print("Device:", device)
    
    # Move model to the selected device - gpu
    global_model.to(device)

    # Configure training components
    # setup the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # setup the optimizer with the learning rate
    model_optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # Learning rate scheduler to reduce LR at specific epochs
    model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer,
                                                milestones=[100, 150], gamma=0.1)

    print("len of train_set::", num_train_instances)
    print("len of val_set::", len(val_loader.dataset))

    # Initialize data structures for client data partitioning
    client_indices = {i: [] for i in range(NUM_CLIENTS)}

    # Initialize a global index pool and digit-specific pools
    global_index_pool = list(range(num_train_instances))
    global_val_pool = list(range(len(val_loader.dataset)))

    client_train_datasets = {}
    client_val_datasets = {}
    client_indices_mapping = {}  # To store the indices allocated to each client

    # Distribute data among clients in a balanced way
    for client_idx in range(NUM_CLIENTS):
        # Filter the global index pool for the current client
        # available_indices = [idx for idx in global_index_pool if train_set[idx][1] not in digit_exclusions[client_idx]]
        available_indices = global_index_pool
        available_val_indices = global_val_pool

        # Calculate how many samples to allocate to this client from remaining pool
        amount_client_samples = len(available_indices) // (NUM_CLIENTS - client_idx)
        amount_client_val_samples = len(available_val_indices) // (NUM_CLIENTS - client_idx)

        # Sample indices randomly for this client
        sampled_indices = random.sample(available_indices, amount_client_samples)
        sampled_val_indices = random.sample(available_val_indices, amount_client_val_samples)
        
        # Create client-specific datasets using indices
        client_train_datasets[client_idx] = Subset(train_loader.dataset, sampled_indices)
        client_val_datasets[client_idx] = Subset(val_loader.dataset, sampled_val_indices)
        client_indices_mapping[client_idx] = sampled_indices
        
        # Remove allocated indices from global pools to avoid duplicates
        global_index_pool = [idx for idx in global_index_pool if idx not in sampled_indices]
        global_val_pool = [idx for idx in global_val_pool if idx not in sampled_val_indices]

    # Create DataLoaders for client datasets
    client_loaders = {
        client_idx: DataLoader(dataset=dataset, batch_size=16, shuffle=True)
        for client_idx, dataset in client_train_datasets.items()
    }

    client_val_loader = {
        client_idx: DataLoader(dataset=dataset, batch_size=16, shuffle=True)
        for client_idx, dataset in client_val_datasets.items()
    }

    # Print statistics about data distribution
    client_data_sizes = [len(client_train_datasets[i]) for i in range(NUM_CLIENTS)]
    print("Client data sizes:", client_data_sizes)

    # Create deep copy of the model for federation
    global_model = copy.deepcopy(model) 

    # Define Flower client for federated learning
    class FlowerClient(flwr.client.NumPyClient):
        """
        Flower client implementation for federated learning.
        
        This class handles the local training, evaluation, and parameter transfer
        for each client in the federated learning system.
        """
        def __init__(self, model, train_loader, val_loader, device):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.losses = []
            

        def get_parameters(self, config):
            """Extract model parameters as NumPy arrays."""
            return [val.cpu().numpy() for val in self.model.state_dict().values()]

        def set_parameters(self, parameters):
            """Update local model with provided parameters."""
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            """Train the local model on client data."""
            self.set_parameters(parameters)
            loss_history = train_local_model(self.model, self.train_loader, loss_fn, 
                            optim.Adam(self.model.parameters(), lr=3e-4), 
                            self.device, epochs=NUM_EPOCHS_PER_CLIENT)
            self.losses.append(loss_history)
            return self.get_parameters(config={}), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            """Evaluate the model on local validation data."""
            self.set_parameters(parameters)
            accuracy, loss, f1 = evaluate_model(self.model, self.val_loader, loss_fn, self.device)

            return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy),
                                                               "f1": float(f1)}        

# Client factory function for Flower simulation
    def client_fn(cid: str) -> Client:
            """Create and return a Flower client for the given client ID."""
            client_model = copy.deepcopy(model)
            train_loader = client_loaders[int(cid)]
            return FlowerClient(client_model, train_loader, val_loader, device).to_client()
    
    # Create Flower client application
    client =  ClientApp(client_fn=client_fn)
    
    # Define server-side components function
    def server_fn(context: Context)-> ServerAppComponents:
        """Configure and return server components for federated learning."""
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(strategy=strategy, config=config)

    # Create Flower server application
    server = ServerApp(server_fn=server_fn)

    # Run the federated learning simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS
    )

    # Load the final model from the parameters of the last round
    trained_model = load_model_from_parameters(model) 

    return trained_model


if __name__ == "__main__":
    # Command-line argument parser for different federated learning strategies
    parser = argparse.ArgumentParser(description='Federated Learning with multiple strategies')
    parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam', 'fedtrimmedavg', 'fedprox'], default='fedavg',
                        help='Choose federated learning strategy (default: fedavg)')
    parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
    parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

    args = parser.parse_args()

    # Configure Vision Transformer model hyperparameters
    image_size = 28
    embed_dim=256
    hidden_dim=embed_dim*3
    num_heads=8
    num_layers=6
    patch_size=7
    num_patches=16
    num_channels=1
    dropout=0.2
    num_classes = 2

    # Initialize Vision Transformer model
    model = VisionTransformer(embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            patch_size=patch_size,
                            num_channels=num_channels,
                            num_patches=num_patches,
                            num_classes=num_classes,
                            dropout=dropout)

    # Run the federated learning process with the configured model
    federated_learning_process(model=model)