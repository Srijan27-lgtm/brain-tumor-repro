import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import timm

# Import our unified data pipeline and model architecture
from data_pretrained import PretrainedDataset, CLASS_TO_IDX
from models import NeXtSwinX

print("="*70)
print("🚀 NeXtSwin-X TRAINING SCRIPT (Table 3 Reproduction)")
print("="*70)

# 1. SETUP & CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paper: "All models were trained under an identical, abbreviated protocol of 10 epochs"
EPOCHS = 10 
BATCH_SIZE = 32

# Set seed for reproducibility (Change to 43, 44 for the other runs)
SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ds_name = "srijan2715/sartaj-final-split" # Change to "srijan2715/brisc2025-paper-pipeline" for BRISC
hf_token = os.environ.get("HF_TOKEN")

# 2. LOAD DATA
print("\n📥 Loading datasets...")
raw_train = load_dataset(ds_name, split="train", token=hf_token)
raw_val = load_dataset(ds_name, split="val", token=hf_token)

train_ds = PretrainedDataset(raw_train, is_training=True)
val_ds = PretrainedDataset(raw_val, is_training=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# 3. MODEL INSTANTIATION
print("\n🏗️ Building NeXtSwin-X Architecture...")
model = NeXtSwinX(num_classes=4).to(DEVICE)

# 4. DIFFERENTIAL LEARNING RATE (DLR) SETUP (Table 2)
# Paper: CNN backbone (3e-5), Transformer backbone (5e-5), Head/Fusion (1e-4)
# Paper: Weight decay (3e-5 for backbones, up to 1e-2 for head)
param_groups = [
    {'params': model.cnn_backbone.parameters(), 'lr': 3e-5, 'weight_decay': 3e-5},
    {'params': model.swin_backbone.parameters(), 'lr': 5e-5, 'weight_decay': 3e-5},
    {'params': [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and 'swin_backbone' not in n], 
     'lr': 1e-4, 'weight_decay': 1e-2}
]

optimizer = optim.AdamW(param_groups)

# Paper: "Cosine Annealing scheduler... decreases... to a minimum value of 1e-6"
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Paper: "Cross-Entropy Loss with a label smoothing factor of 0.1"
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Paper: "Automatic Mixed Precision (AMP)"
scaler = torch.cuda.amp.GradScaler()

# 5. TRAINING LOOP
print(f"\n🏋️ Training for {EPOCHS} epochs (Seed: {SEED})...")
logs = []
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        x = batch['image'].to(DEVICE, non_blocking=True)
        y = batch['label'].to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        # AMP Forward Pass
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        
        # AMP Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)
    
    train_acc = train_correct / train_total
    train_loss = train_loss / train_total
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['image'].to(DEVICE, non_blocking=True)
            y = batch['label'].to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                out = model(x)
            _, preds = torch.max(out, 1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
    
    val_acc = val_correct / val_total
    
    # Update Scheduler
    scheduler.step()
    
    # Log
    logs.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save Best Checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'nextswinx_seed{SEED}_best.pth')
        print(f"   💾 Saved new best model (Val Acc: {best_val_acc:.4f})")

# 6. SAVE ARTIFACTS
pd.DataFrame(logs).to_csv(f'training_log_seed{SEED}.csv', index=False)
print(f"\n✅ Saved training_log_seed{SEED}.csv")
print(f"✅ Final Best Validation Accuracy: {best_val_acc:.4f}")
print("="*70)
print("🏁 TRAINING COMPLETE.")
print("="*70)
