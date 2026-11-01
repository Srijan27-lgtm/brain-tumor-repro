import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Import our single source of truth
from data import SartajDataset, CLASS_TO_IDX

print("="*70)
print("🚀 SARTAJ SCRATCHCNN TRAINING SCRIPT")
print("="*70)

# 1. SETUP & CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

ds_name = "srijan2715/sartaj-final-split"
# Securely load token from environment variable
hf_token = os.environ.get("HF_TOKEN") 

# 2. LOAD DATA
print("\n📥 Loading datasets...")
raw_train = load_dataset(ds_name, split="train", token=hf_token)
raw_val = load_dataset(ds_name, split="val", token=hf_token)

train_ds = SartajDataset(raw_train, is_training=True)
val_ds = SartajDataset(raw_val, is_training=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# 3. MODEL ARCHITECTURE
class ScratchCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ScratchCNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = ScratchCNN(num_classes=4).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 4. TRAINING LOOP
print("\n🏋️ Training for 15 epochs...")
logs = []
best_val_acc = 0.0
best_model_state = None
patience = 0
patience_limit = 5

for epoch in range(15):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        x = batch['image'].to(DEVICE)
        y = batch['label'].to(DEVICE)
        
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)
    
    train_acc = train_correct / train_total
    train_loss = train_loss / train_total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['image'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            out = model(x)
            _, preds = torch.max(out, 1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
    
    val_acc = val_correct / val_total
    
    # Log
    logs.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
    print(f"Epoch {epoch+1}/15 | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    
    # Save best model based on VAL accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
        
    if patience >= patience_limit:
        print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
        break

# 5. SAVE ARTIFACTS
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"\n✅ Loaded best model (Val Acc: {best_val_acc:.4f})")

# Save CSV and Curves
pd.DataFrame(logs).to_csv("training_log.csv", index=False)
df = pd.DataFrame(logs)
plt.figure(figsize=(8, 4))
plt.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
plt.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Curves (Best Checkpoint from Val)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.close()
print("✅ Saved training_log.csv and training_curves.png")

# Save the final verified weights
torch.save(model.state_dict(), 'sartaj_seed42_FINAL.pth')
print("✅ Saved sartaj_seed42_FINAL.pth")
print("\n" + "="*70)
print("🏁 TRAINING COMPLETE.")
print("="*70)