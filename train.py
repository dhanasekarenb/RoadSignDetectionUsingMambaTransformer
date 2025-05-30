import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import traceback
from dataset_loader import get_dataloaders
from mamba_model import MambaTransformerClassifier

# Suppress only FutureWarnings & UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self, config):
        try:
            print("📂 Loading dataset...")
            self.train_loader, self.val_loader = get_dataloaders(batch_size=config['batch_size'])
            print(f"✅ Dataset loaded successfully: {len(self.train_loader)} train batches, {len(self.val_loader)} validation batches.")
        except Exception as e:
            print("❌ Dataset loading failed:", e)
            print(traceback.format_exc())
            raise

        try:
            print("🛠 Initializing model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = MambaTransformerClassifier().to(self.device)
            print("✅ Model initialized on", self.device)
        except Exception as e:
            print("❌ Model initialization failed:", e)
            print(traceback.format_exc())
            raise

        try:
            print("⚙️ Setting up optimizer and loss function...")
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
            self.scaler = torch.amp.GradScaler()
            print("✅ Optimizer and loss function ready.")
        except Exception as e:
            print("❌ Optimizer setup failed:", e)
            print(traceback.format_exc())
            raise

        # Track best validation accuracy
        self.best_val_acc = 0.0
        self.best_train_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        try:
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        except Exception as e:
            print("❌ Fatal Error in Training Loop:", e)
            print(traceback.format_exc())
            raise

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        try:
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

        except Exception as e:
            print("❌ Fatal Error in Validation Loop:", e)
            print(traceback.format_exc())
            raise

        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        return val_loss, val_acc

    def train(self, epochs):
        print(f"🚀 Training for {epochs} epochs on {self.device}...\n")

        for epoch in range(epochs):
            try:
                print(f"📌 Epoch {epoch + 1}/{epochs}")

                train_loss, train_acc = self.train_epoch()
                print(f"📉 Train Loss: {train_loss:.4f} | 🎯 Train Acc: {train_acc:.2f}%")

                val_loss, val_acc = self.validate_epoch()
                print(f"🔍 Validation Loss: {val_loss:.4f} | 🎯 Validation Acc: {val_acc:.2f}%")

                # Save model if validation accuracy improves
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_train_acc = train_acc
                    self.best_epoch = epoch + 1
                    print(f"💾 Saving best model with accuracy {val_acc:.2f}%...")
                    torch.save(self.model.state_dict(), "mamba_transformer_best.pth")

            except Exception as e:
                print(f"❌ Fatal Error in Epoch {epoch+1}:", e)
                print(traceback.format_exc())
                raise

        print("\n🎯 **Training Summary** 🎯")
        print(f"🏆 Best Epoch: {self.best_epoch}")
        print(f"📈 Best Train Acc: {self.best_train_acc:.2f}%")
        print(f"🔍 Best Validation Acc: {self.best_val_acc:.2f}%")

if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'lr': 0.001,
        'epochs': 15
    }

    try:
        trainer = Trainer(config)
        trainer.train(config['epochs'])
    except Exception as e:
        print("❌ Fatal error in main execution:", e)
        print(traceback.format_exc())
