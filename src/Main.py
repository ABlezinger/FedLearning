import SavingStrategy
import argparse
import copy
import json
import os 
import flwr
import torch
import numpy as np
from model_utils import VisionTransformer, evaluate_model
from dataPreparation import get_BreastMNIST, get_PneumoniaMNIST, get_ChestMNIST
from fedLearning import federated_learning_process
from continualUtils import calculate_backwards_transfer, calculate_forward_transfer

parser = argparse.ArgumentParser(description='Federated Learning with multiple strategies')
parser.add_argument('--strategy', type=str, choices=['fedavg', 'fedyogi', 'fedadam', 'fedtrimmedavg', 'fedprox'], default='fedavg',
                    help='Choose federated learning strategy (default: fedavg)')
parser.add_argument('--eta', type=float, default=0.01, help='Server-side learning rate for FedYogi (default: 0.01)')
parser.add_argument('--eta_l', type=float, default=0.0316, help='Client-side learning rate for FedYogi (default: 0.0316)')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for FedYogi (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for FedYogi (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, help='Tau parameter for FedYogi (default: 0.001)')

args = parser.parse_args()
os.makedirs(f"assets/models", exist_ok=True)
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

# initialize global model

global_model = VisionTransformer(embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        patch_size=patch_size,
                        num_channels=num_channels,
                        num_patches=num_patches,
                        num_classes=2,
                        dropout=dropout)

    
# create custom strategy for the federated learning 
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


# set up continual learning setting 
dataset_step_list = ["PneumoniaMNIST", "BreastMNIST", "ChestMNIST"]

_, _, testP, _ = get_PneumoniaMNIST()
_, _, testB, _ = get_BreastMNIST()
_, _, testC, _ = get_ChestMNIST()
test_sets = [testP, testB, testC]

baseline_vector_acc = []
baseline_vector_f1 = []

for i, dataset in enumerate(dataset_step_list):
    expert_model = copy.deepcopy(global_model)
    expert_model = federated_learning_process(
            model=expert_model, 
            dataset=dataset,
            strategy=strategy, 
            clients=NUM_CLIENTS, 
            rounds=NUM_ROUNDS, 
            epochs=NUM_EPOCHS,
        )
    acc, _ , f1 = evaluate_model(expert_model, test_sets[i])
    baseline_vector_acc.append(acc)
    baseline_vector_f1.append(f1)



test_matrix_acc = np.zeros((len(test_sets), len(test_sets)))
test_matrix_f1 = np.zeros((len(test_sets), len(test_sets)))

# continual learning loop
for step, dataset in enumerate(dataset_step_list):
    print(f"training model on step {step} with dataset {dataset}...")

    global_model = federated_learning_process(
        model=global_model, 
        dataset=dataset,
        strategy=strategy, 
        clients=NUM_CLIENTS, 
        rounds=NUM_ROUNDS, 
        epochs=NUM_EPOCHS,
    )
    for i in range(step + 1):
        acc, _, f1 = evaluate_model(global_model, test_sets[i])
        test_matrix_acc[step][i] = acc
        test_matrix_f1[step][i] = f1

    # Compute forgetting, forward transfer...
fwt_acc = calculate_forward_transfer(baseline_vector_acc, test_matrix_acc)
fwt_f1 = calculate_forward_transfer(baseline_vector_f1, test_matrix_f1)
bwt_acc = calculate_backwards_transfer(test_matrix_acc)
bwt_f1 = calculate_backwards_transfer(test_matrix_f1)

results = {
    "baseline_acc": baseline_vector_acc,
    "baseline_f1": baseline_vector_f1,
    "test_results_acc": test_matrix_acc[len(dataset_step_list) - 1].tolist(),
    "test_results_f1": test_matrix_f1[len(dataset_step_list) - 1].tolist(),
    "fwd_acc": fwt_acc,
    "bwt_acc": bwt_acc,
    "fwt_f1": fwt_f1,
    "bwt_f1": bwt_f1
}

os.makedirs(f"results/{args.strategy}", exist_ok=True)
json.dump(results, open(f"results/{args.strategy}/{NUM_CLIENTS}_Clients_{NUM_ROUNDS}_Rounds_results.json", "w"))

