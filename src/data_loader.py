import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader, Subset

def load_data(data_dir, batch_size=64, num_samples=None):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    if num_samples:
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader, dataset.classes
