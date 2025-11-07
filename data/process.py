import os 
import shutil
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

data_path =r"H:\python\Driver_drowsiness_mlops\data\processed"

data_dir = r"H:\python\Driver_drowsiness_mlops\data\raw\drowsiness-detection"
train_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\train"
val_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\val"
classes =["closed_eye","open_eye"]


# Create folders
for split in [train_dir, val_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# Split ratio
split_ratio = 0.8

# Move images
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    for img in train_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

def get_loaders(batch_size=32):

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
    
return train_loader,val_loader

for images, labels in train_loader:
    print(f"Images batch shape: {images.shape}")  # [batch_size, channels, H, W]
    print(f"Labels batch shape: {labels.shape}")  # [batch_size]
    break  # just test one batch
