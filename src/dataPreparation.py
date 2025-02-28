import os 
import sys
from medmnist import BreastMNIST, DermaMNIST, PneumoniaMNIST, ChestMNIST, TissueMNIST
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define project and data paths
project_path = sys.path[0]
data_path = project_path + "/data"

# Define preprocessing transformation for MNIST datasets
MNIST_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

def get_BreastMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load and prepare the BreastMNIST dataset.
    
    Creates directory for data if it doesn't exist, downloads the dataset,
    applies preprocessing, and creates data loaders.
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_classes: Number of classes in the dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Load train, validation and test datasets with preprocessing
    train = BreastMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = BreastMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = BreastMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    # Create data loaders with batch size 16
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    # Calculate number of unique classes
    num_classes = len(np.unique(train.labels))
    
    return train_loader, val_loader, test_loader, num_classes

def get_PneumoniaMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load and prepare the PneumoniaMNIST dataset.
    
    Creates directory for data if it doesn't exist, downloads the dataset,
    applies preprocessing, and creates data loaders.
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_classes: Number of classes in the dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Load train, validation and test datasets with preprocessing
    train = PneumoniaMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = PneumoniaMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = PneumoniaMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)

    # Create data loaders with batch size 16
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    # Calculate number of unique classes
    num_classes = len(np.unique(train.labels))

    return train_loader, val_loader, test_loader, num_classes

def get_ChestMNIST() -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load and prepare the ChestMNIST dataset, converting multi-label to binary classification.
    
    Creates directory for data if it doesn't exist, downloads the dataset,
    applies preprocessing, converts multi-label to binary, and creates data loaders.
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_classes: Number of classes in the dataset (2 after binary conversion)
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    def multi_to_binary_chest(dataset: Dataset):
        """
        Convert multi-label ChestMNIST to binary classification.
        
        If any of the labels are positive (sum > 0), assigns label 1, otherwise 0.
        
        Args:
            dataset: ChestMNIST dataset with multi-label format
            
        Returns:
            dataset: Same dataset with binary labels
        """
        new_lables = ar = np.array([np.where(np.sum(row) > 0, 1, 0) for row in dataset.labels]).reshape(-1, 1)
        dataset.labels = new_lables
        return dataset

    # Load train, validation and test datasets with preprocessing
    train = ChestMNIST(split="train", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    train = multi_to_binary_chest(train)
    val = ChestMNIST(split="val", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    val = multi_to_binary_chest(val)
    test = ChestMNIST(split="test", download=True, root=project_path + "/data", transform=MNIST_preprocess)
    test = multi_to_binary_chest(test)

    # Create data loaders with batch size 16
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)

    # Calculate number of unique classes (should be 2 after binary conversion)
    num_classes = len(np.unique(train.labels))

    return train_loader, val_loader, test_loader, num_classes

# Script execution section for testing
if __name__ == "__main__":
    # Test the ChestMNIST function and print results
    t, v, test, n = get_ChestMNIST()

    print(t.dataset.labels)     # Print the converted binary labels
    print(n)                    # Print number of classes