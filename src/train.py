import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

train_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\train"
val_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\val"

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = ImageFolder(root=val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    print(f"Images batch shape: {images.shape}")  # [batch_size, channels, H, W]
    print(f"Labels batch shape: {labels.shape}")  # [batch_size]
    break  # just test one batch
