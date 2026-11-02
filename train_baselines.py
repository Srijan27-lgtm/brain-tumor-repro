import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from huggingface_hub import HfApi, create_repo

# Import our unified data pipeline and baseline models
from data_pretrained import PretrainedDataset, CLASS_TO_IDX
from models_baselines import (
    ConvNeXtOnly, SwinOnly, ResNet50Only, EfficientNetV2Only, 
    BiFNN_ResNet50, TwoStream_Spatial_ResNet50
)

# ==============================================================================
# 1. MODEL REGISTRY
# ==============================================================================
MODEL_REGISTRY = {
    'convnext': ConvNeXtOnly,
    'swin': SwinOnly,
    'resnet50': ResNet50Only,
    'efficientnet': EfficientNetV2Only,
    'bifnn': BiFNN_ResNet50,
    'twostream': TwoStream_Spatial_ResNet50
}

def run_training(model_name, seed, dataset_name):
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {model_name.upper()} | Seed: {seed} | Dataset: {dataset_name.split('/')[-1]}")
    print(f"{'='*70}")

    # 1. FORCE CUDA. If GPU is missing, crash immediately. NO CPU FALLBACK.
    assert torch.cuda.is_available(), "❌ CUDA is not available! Script requires a GPU."
    DEVICE = torch.device("cuda")
    print(f"🔥 FORCED DEVICE: {DEVICE} ({torch.cuda.get_device_name(0)})")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    hf_token = os.environ.get("HF_TOKEN")

    # Data Loading
    print("📥 Loading dataset...")
    raw_train = load_dataset(dataset_name, split="train", token=hf_token)
    raw_val = load_dataset(dataset_name, split="val", token=hf_token)
    
    train_ds = PretrainedDataset(raw_train, is_training=True)
    val_ds = PretrainedDataset(raw_val, is_training=False)
    
    # num_workers=0 is CRITICAL for cloud containers to prevent shared memory bottlenecks
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Model Instantiation
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=4).to(DEVICE)
    
    # SANITY CHECK: Prove the model is on the GPU
    print(f"✅ MODEL IS ON: {next(model.parameters()).device}")

    # Differential Learning Rate (2 Groups for Baselines)
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': 3e-5, 'weight_decay': 3e-5},
        {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-2}
    ]

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # MODERN PYTORCH AMP SYNTAX (Fixes deprecation warnings and ensures GPU usage)
    scaler = torch.amp.GradScaler('cuda')

    # Training Loop
    logs = []
    best_val_acc = 0.0

    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch in train_loader:
            x, y = batch['image'].to(DEVICE, non_blocking=True), batch['label'].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            # MODERN AUTOCAST
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['image'].to(DEVICE, non_blocking=True), batch['label'].to(DEVICE, non_blocking=True)
                # MODERN AUTOCAST
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(x)
                _, preds = torch.max(out, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step()

        logs.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        print(f"Epoch {epoch+1}/10 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Save locally
    pth_filename = f"{model_name}_{dataset_name.split('/')[-1]}_seed{seed}_best.pth"
    csv_filename = f"logs_{model_name}_{dataset_name.split('/')[-1]}_seed{seed}.csv"
    
    torch.save(model.state_dict(), pth_filename)
    pd.DataFrame(logs).to_csv(csv_filename, index=False)
    print(f"💾 Saved locally: {pth_filename}")

    # ==============================================================================
    # FAILSAFE UPLOAD TO HUGGING FACE
    # ==============================================================================
    print(f"\n🚀 Uploading Seed {seed} results to Hugging Face...")
    try:
        api = HfApi(token=hf_token)
        repo_id = "srijan2715/reproductioncheckpoints"
        
        api.upload_file(
            path_or_fileobj=pth_filename,
            path_in_repo=f"baselines/{pth_filename}",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"   ✅ Uploaded {pth_filename} to HF!")
        
        api.upload_file(
            path_or_fileobj=csv_filename,
            path_in_repo=f"baselines/{csv_filename}",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"   ✅ Uploaded {csv_filename} to HF!")
        
    except Exception as e:
        print(f"   ❌ HF Upload failed: {e}. Download manually immediately!")

    print(f"✅ Run Complete! Best Val Acc: {best_val_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    run_training(args.model, args.seed, args.dataset)
