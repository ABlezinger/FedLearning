import os 
import sys
from medmnist import BreastMNIST, DermaMNIST, PneumoniaMNIST, ChestMNIST, TissueMNIST
import torchvision
from torch.utils.data import DataLoader
import numpy as np

project_path = sys.path[0]
data_path = project_path + "/data"

MNIST_preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5,), (0.5,))])

def get_BreastMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    train = BreastMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = BreastMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = BreastMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    num_classes = len(np.unique(train.labels))
    
    return train_loader, val_loader, test_loader, num_classes

def get_DermaMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train = DermaMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = DermaMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = DermaMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    num_classes = len(np.unique(train.labels))
    
    return train_loader, val_loader, test_loader, num_classes

def get_PneumoniaMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train = PneumoniaMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = PneumoniaMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = PneumoniaMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    num_classes = len(np.unique(train.labels))
    return train_loader, val_loader, test_loader, num_classes

def get_ChestMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    def multi_to_binary_chest(dataset: Dataset):
        labels = dataset.labels
        new_lables = ar = np.array([np.where(np.sum(row) > 0, 1, 0) for row in dataset.labels]).reshape(-1, 1)
        dataset.labels = new_lables
        return dataset

    train = ChestMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    train = multi_to_binary_chest(train)
    val = ChestMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = multi_to_binary_chest(val)
    test = ChestMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = multi_to_binary_chest(test)



    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    num_classes = len(np.unique(train.labels))

    return train_loader, val_loader, test_loader, num_classes
  
def get_TissueMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train = TissueMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = TissueMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = TissueMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    num_classes = len(np.unique(train.labels))
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    t, v, test, n = get_ChestMNIST()

    print(t.dataset.labels)
    print(n)