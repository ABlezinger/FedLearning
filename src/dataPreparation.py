import os 
import sys
from medmnist import BreastMNIST
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

project_path = sys.path[0]
data_path = project_path + "\data"

MNIST_preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5,), (0.5,))])

def get_BreastMNIST():

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    train_breast = BreastMNIST(split="train", download=True, root=project_path + "\data", transform=MNIST_preprocess)
    val_breast = BreastMNIST(split="val", download=True, root=project_path + "\data", transform=MNIST_preprocess)
    test_breast = BreastMNIST(split="test", download=True, root=project_path + "\data", transform=MNIST_preprocess)

    train_loader = DataLoader(train_breast, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_breast, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_breast, batch_size=16, shuffle=True)
    
    return train_loader, val_loader, test_loader