from medmnist import ChestMNIST
import numpy as np

# Load the ChestMNIST dataset
chestmnist = ChestMNIST(split="train", download=True)
images = chestmnist.imgs
labels = chestmnist.labels

# Convert images to the correct format (num_samples, 1, 28, 28) for PyTorch
images = np.expand_dims(images, axis=1).astype(np.float32) / 255.0
labels = labels.astype(np.float32)

selected_labels_index = 2

binary_labels = labels[:, selected_labels_index]
binary_labels = np.where(binary_labels == 1, 1, 0)

unique_classes = np.unique(binary_labels)
print("Possible classes in binary setting:", unique_classes)