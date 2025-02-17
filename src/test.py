from models import VisionTransformer
from SavingStrategy import load_params
import torch




image_size = 28
embed_dim=256
hidden_dim=embed_dim*3
num_heads=8
num_layers=6
patch_size=7
num_patches=16
num_channels=1
num_classes=2
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

weights = load_params()

with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.copy_(weight)

print(model)