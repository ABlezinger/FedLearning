import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
# from datasets import load_dataset
from medmnist import BreastMNIST
import copy
from torch.utils.data import DataLoader, Dataset, Subset

project_path = sys.path[0]
data_path = project_path + "\data"

random.seed(42)

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
NUM_IMAGES = 3
# examples = torch.stack([val_breast[idx][0] for idx in range(NUM_IMAGES)], dim=0)
# img_grid = torchvision.utils.make_grid(examples, nrow=2, normalize=True, pad_value=0.9)
# img_grid = img_grid.permute(1, 2, 0)


# plt.figure(figsize=(8, 8))
# plt.title("Image examples of the MedMNIST dataset")
# plt.imshow(img_grid)
# plt.axis("off")
# plt.show()
# plt.close()

# TODO evtl dynamisch machen (num_patches , size etc.)
image_size = 28
embed_dim=256
hidden_dim=embed_dim*3
num_heads=8
num_layers=6
patch_size=7
num_patches=16
num_channels=1
num_classes=1
dropout=0.2



def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape # [B, C, H, W], MNIST [B, 1, 28, 28]
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size) # [B, C, H', p_H, W', p_W], MNIST [B, 1, 4, 7, 4, 7]
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W], MNIST [B, 4, 4, 1, 7, 7]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W], MNIST [B, 16, 1, 7, 7]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W], MNIST [B, 16, 49]
    return x


# Visualize the image patches
# img_patches = img_to_patch(examples, patch_size=patch_size, flatten_channels=False)

# fig, ax = plt.subplots(examples.shape[0], 1, figsize=(14, 12))
# fig.suptitle("Images as input sequences of patches")
# for i in range(examples.shape[0]):
#     img_grid = torchvision.utils.make_grid(img_patches[i], nrow=int(image_size/patch_size), normalize=True, pad_value=0.9)
#     img_grid = img_grid.permute(1, 2, 0)
#     ax[i].imshow(img_grid)
#     ax[i].axis("off")
# plt.show()
# plt.close()


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB or 1 for grayscale)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)        # x.shape ---> batch, num_patches, (patch_size**2)
        B, T, _ = x.shape
        x = self.input_layer(x)                     # x.shape ---> batch, num_patches, embed_dim

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)        # x.shape ---> batch, num_patches+1, embed_dim
        x = x + self.pos_embedding[:, : T + 1]      # x.shape ---> batch, num_patches+1, embed_dim

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)                       # x.shape ---> num_patches+1, batch, embed_dim
        x = self.transformer(x)                     # x.shape ---> num_patches+1, batch, embed_dim

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
    

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
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCELoss()
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
            # _, predicted = torch.max(outputs.data, 1)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# Create a function to train locally on a client
def train_local_model(model, train_loader, loss_fn, optimizer, device, epochs):
    model.train()
    loss_history = []
    for epoch in tqdm(range(epochs)):
        losses = [] 
        for imgs, labels in train_loader:
            labels = labels.to(torch.float)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            # print(f"Shape Pred: {preds.shape}")
            # print(f"shape Label {labels.shape}")
            loss = loss_fn(preds, labels)
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

print("len of train_set::", len(test_breast))

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
global_index_pool = list(range(len(train_breast)))

client_datasets = {}
client_indices_mapping = {}  # To store the indices allocated to each client

# for client_idx in range(len(digit_exclusions)):
for client_idx in range(NUM_CLIENTS):
    # Filter the global index pool for the current client
    # available_indices = [idx for idx in global_index_pool if train_set[idx][1] not in digit_exclusions[client_idx]]
    available_indices = global_index_pool

    amount_client_samples = len(available_indices) // (NUM_CLIENTS - client_idx)

    # Sample 10,000 unique indices for the current client
    sampled_indices = random.sample(available_indices, amount_client_samples)
    
    # Assign to the client dataset and store the indices
    client_datasets[client_idx] = Subset(train_breast, sampled_indices)
    client_indices_mapping[client_idx] = sampled_indices
    
    # Remove the sampled indices from the global pool
    global_index_pool = [idx for idx in global_index_pool if idx not in sampled_indices]

# Create DataLoaders for each client
client_loaders = {
    client_idx: DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for client_idx, dataset in client_datasets.items()
}

# Calculate and print the length of each client's dataset
client_data_sizes = [len(client_datasets[i]) for i in range(NUM_CLIENTS)]
print("Client data sizes:", client_data_sizes)

# # Print the indices allocated to each client
# for client_idx, indices in client_indices_mapping.items():
#     print(f"Client {client_idx} has indices: {indices[:20]} ...")  # Printing first 20 indices for brevity
#     print(f"Total indices for Client {client_idx}: {len(indices)}")

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
num_rounds = 10  # Number of communication rounds
epochs_per_client = 20  # Number of local epochs per client


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
        accuracy=evaluate_model(client_models[client_idx], test_loader, device)
        print("accuracy:",accuracy)
    
    # Aggregate client models to update global model
    federated_averaging(global_model, client_models,client_data_sizes)


result = evaluate_model(global_model, test_loader, device)

print(f"RESULT: {result}")

