import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.process import get_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn import SimpleCNN
import mlflow
import mlflow.pytorch

# -----------------------------
# Load data
# -----------------------------
train_loader, val_loader = get_loaders(batch_size=32)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15
    patience = 3

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # -----------------------------
    # MLflow tracking
    # -----------------------------
    with mlflow.start_run():
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_fn", "BCELoss")

        for epoch in range(num_epochs):
            # ---------- Training ----------
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # ---------- Validation ----------
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * correct / total

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # ---------- MLflow metrics ----------
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            # ---------- Save checkpoint ----------
            os.makedirs(r"H:\python\Driver_drowsiness_mlops\Outputs\models", exist_ok=True)
            torch.save(model.state_dict(), rf"H:\python\Driver_drowsiness_mlops\Outputs\models_epoch{epoch+1}.pth")

            # ---------- Early stopping ----------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), rf"H:\python\Driver_drowsiness_mlops\Outputs\best_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

        # ---------- Log best model ----------
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Optional: log plot if available
        # plot_path = "path_to_plot.png"  # define your plot path
        # mlflow.log_artifact(plot_path)

        mlflow.pytorch.log_model(model, "model")

    print("✅ Training complete and best model logged to MLflow.")

def plot_metrics(train_losses, val_losses, val_accuracies, save_path="H:\python\Driver_drowsiness_mlops\Outputs\figures\metrics_plot.png"):
    plt.figure(figsize=(12, 5))

    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Metrics plot saved at: {save_path}")
    plt.close()

if __name__ == "__main__":
    train_model()
