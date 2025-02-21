
import SavingStrategy
import argparse
import copy
from src.model_utils import VisionTransformer
from fedLearning import federated_learning_process
import flwr


parser = argparse.ArgumentParser(description='Federated Learning with multiple strategies')
parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam', 'fedtrimmedavg', 'fedprox'], default='fedavg',
                    help='Choose federated learning strategy (default: fedavg)')
parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

args = parser.parse_args()

image_size = 28
embed_dim=256
hidden_dim=embed_dim*3
num_heads=8
num_layers=6
patch_size=7
num_patches=16
num_channels=1
dropout=0.2

NUM_CLIENTS = 3
NUM_ROUNDS = 5
NUM_EPOCHS = 10


global_model = None
if global_model is None:
    global_model = VisionTransformer(embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            num_layers=num_layers,
                            patch_size=patch_size,
                            num_channels=num_channels,
                            num_patches=num_patches,
                            num_classes=2,
                            dropout=dropout)

    

if args.strategy == 'fedyogi':
    # Initialize FedYogi strategy
    strategy = SavingStrategy.SaveModelYogi(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        eta=args.eta,
        eta_l=args.eta_l,
        beta_1=args.beta1,
        beta_2=args.beta2,
        tau=args.tau
    )
        
elif args.strategy == 'fedadam':
    strategy = SavingStrategy.SaveModelAdam(
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

elif args.strategy == 'fedprox':
    strategy = SavingStrategy.SaveModelProx(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        proximal_mu=0.01                 # Proximal term parameter (controls how far local models can deviate)
    )

elif args.strategy == 'fedtrimmedavg':
    strategy = SavingStrategy.SaveModelTrimmedAvg(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,     # Für deine 3 Clients
        min_available_clients=NUM_CLIENTS,  
        fraction_fit=1.0,                # Alle verfügbaren Clients nutzen
        fraction_evaluate=1.0,           # Alle Clients für Evaluation nutzen
        beta=0.1                         # 10% von beiden Enden trimmen
    )

else:  
    strategy = SavingStrategy.SaveModelFedAvg(
        initial_parameters=flwr.common.ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()]),
        min_fit_clients=NUM_CLIENTS,     # Für deine 3 Clients
        min_available_clients=NUM_CLIENTS,  
        fraction_fit=1.0,                # Alle verfügbaren Clients nutzen
        fraction_evaluate=1.0,           # Alle Clients für Evaluation nutzen
        #beta=0.1                         # 10% von beiden Enden trimmen
    )

trained_new_model = federated_learning_process(
     model=global_model, 
     dataset="PneumoniaMNIST",
     strategy=strategy, 
     clients=NUM_CLIENTS, 
     rounds=NUM_ROUNDS, 
     epochs=NUM_EPOCHS, 
)
