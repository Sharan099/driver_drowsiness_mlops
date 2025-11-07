import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import process
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cnn import CNNModel
import torch.optim as optim

print(os.getcwd())
print(sys.path)
'''
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15  # can increase since we have early stopping
patience = 3     # stop if validation loss doesn't improve for 3 epochs

# -----------------------------
# 3️⃣ Training loop with early stopping
# -----------------------------
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in process.train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(process.train_loader)
    train_losses.append(train_loss)
    
    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in process.val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(process.val_loader)
    val_losses.append(val_loss)
    
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model.state_dict(), rf"H:\python\Driver_drowsiness_mlops\Outputs\models{epoch+1}.pth")
    
    # -----------------------------
    # Early stopping
    # -----------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(rf"Early stopping triggered after {epoch+1} epochs.")
            break

# -----------------------------
# 4️⃣ Plot training & validation loss
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(rf"H:\python\Driver_drowsiness_mlops\Outputs\trainingvsval{epoch}.png")
plt.show()
'''