import os
import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from sklearn.metrics import f1_score

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Implements the Attention Block for the Vision Transformer.

        Args:
            embed_dim (int): Dimensionality of input and attention feature vectors.
            hidden_dim (int): Dimensionality of the hidden layer in the feed-forward network.
            num_heads (int): Number of attention heads in the Multi-Head Attention block.
            dropout (float): Dropout rate applied in the feed-forward network.
        """
        super().__init__()
        
        # Layer normalization for input stabilization
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        # Multi-Head Attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        # Layer normalization before the feed-forward network
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        # Feed-forward network with GELU activation and dropout
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward pass through the Attention Block."""
        inp_x = self.layer_norm_1(x)
        # Apply Multi-Head Attention and residual connection
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # Apply feed-forward network and residual connection
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
        Implements the Vision Transformer (ViT) model.

        Args:
            embed_dim (int): Dimensionality of input feature vectors.
            hidden_dim (int): Dimensionality of hidden layers in the Transformer.
            num_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            num_heads (int): Number of attention heads in each Attention Block.
            num_layers (int): Number of Attention Blocks in the Transformer.
            num_classes (int): Number of output classes for classification.
            patch_size (int): Size of each image patch.
            num_patches (int): Maximum number of patches per image.
            dropout (float): Dropout rate applied in the Transformer.
        """
        super().__init__()

        self.patch_size = patch_size

        # Input layer to project patches into the embedding dimension
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        # Stack of Attention Blocks
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, embed_dim//2), 
            nn.ReLU(), 
            nn.Linear(embed_dim//2, num_classes)
            )
        self.dropout = nn.Dropout(dropout)

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        """Forward pass through the Vision Transformer."""
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
    
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Converts an image into patches.

    Args:
        x (torch.Tensor): Input image tensor of shape [B, C, H, W].
        patch_size (int): Size of each patch.
        flatten_channels (bool): If True, flattens the patches into feature vectors.

    Returns:
        torch.Tensor: Tensor of patches with shape [B, num_patches, patch_features].
    """
    B, C, H, W = x.shape # [B, C, H, W], MNIST [B, 1, 28, 28]
    # Reshape image into patches
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size) # [B, C, H', p_H, W', p_W], MNIST [B, 1, 4, 7, 4, 7]
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W], MNIST [B, 4, 4, 1, 7, 7]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W], MNIST [B, 16, 1, 7, 7]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W], MNIST [B, 16, 49]
    return x

def load_model_from_parameters(model):
    """
    Loads model parameters from saved weights.

    Args:
        model (nn.Module): The model to load weights into.

    Returns:
        nn.Module: The model with loaded weights.
    """
    weights = load_params()

    with torch.no_grad():
            for param, weight in zip(model.parameters(), weights):
                param.copy_(weight)

    return model	


def train_local_model(model, train_loader, loss_fn, optimizer, device, epochs):
    """
    Trains the model locally on a client's dataset.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        loss_fn (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to use for training (e.g., "cpu", "cuda").
        epochs (int): Number of training epochs.

    Returns:
        list: History of training losses.
    """
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


def evaluate_model(model, test_loader, loss_fn = None, device = "mps"):
    """
    Evaluates the model on a test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        loss_fn (nn.Module, optional): Loss function for evaluation. Defaults to None.
        device (str): Device to use for evaluation (e.g., "cpu", "cuda").

    Returns:
        tuple: Accuracy, loss, and F1 score.
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if not loss_fn is None:
                loss += loss_fn(outputs, labels.squeeze(-1).long()).item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze(-1).long()).sum().item()
            all_labels.extend(labels.squeeze(-1).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    loss /= len(test_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions)
    return accuracy, loss, f1


def load_params():
    """
    Loads model parameters from a saved file.

    Returns:
        list: List of model weights as torch.Tensor.
    """
    loaded_data = np.load("assets/models/latest_weights.npz")
    weights = [torch.tensor(loaded_data[key]) for key in loaded_data.files]
    return weights