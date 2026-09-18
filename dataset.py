import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    # Member 1: Training transforms with gentle data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Test/Inference transforms: standard normalization only
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Shuffling training data prevents the optimizer from getting stuck in cyclical batch patterns.
    # Testing data doesn't require shuffling, which ensures reproducible evaluations.
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Should be [32, 1, 28, 28]
    print(f"Batch labels shape: {labels.shape}")  # Should be [32]