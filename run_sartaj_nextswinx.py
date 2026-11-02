import os
import subprocess
import sys

print("="*70)
print("🚀 STARTING SARTAJ PIPELINE FOR SEEDS 43 & 44 (SKIPPING 42)")
print("="*70)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 👇 ONLY 43 and 44 👇
SEEDS = [43, 44] 
DATASET_NAME = "srijan2715/sartaj-final-split"  
BASE_SCRIPT = "train_pretrained.py"
TEMP_SCRIPT = "train_sartaj_temp.py"
HF_REPO_ID = "srijan2715/reproductioncheckpoints"

# 👇 PASTE YOUR BRAND NEW HUGGING FACE TOKEN HERE 👇
# Make sure it has "Write" permissions!
HF_TOKEN = "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" 
os.environ["HF_TOKEN"] = HF_TOKEN

print(f"📂 Dataset: {DATASET_NAME}")
print("="*70)

# 1. Read the original script
with open(BASE_SCRIPT, 'r') as f:
    original_code = f.read()

# Optional: Fix the harmless FutureWarnings in the temp script
original_code = original_code.replace(
    "scaler = torch.cuda.amp.GradScaler()", 
    "scaler = torch.amp.GradScaler('cuda')"
)
original_code = original_code.replace(
    "with torch.cuda.amp.autocast():", 
    "with torch.autocast(device_type='cuda', dtype=torch.float16):"
)

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

# 2. Loop through seeds
for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"🔥 PREPARING RUN FOR SEED {seed}...")
    print(f"{'='*70}")
    
    # Modify the code dynamically to use the current seed
    new_code = original_code.replace('SEED = 42', f'SEED = {seed}')
    new_code = new_code.replace('SEED = 43', f'SEED = {seed}') 
    new_code = new_code.replace('SEED = 44', f'SEED = {seed}') 
    
    # Update dataset name
    new_code = new_code.replace('srijan2715/brisc2025-paper-pipeline', DATASET_NAME)
    
    # Write to temporary script
    with open(TEMP_SCRIPT, 'w') as f:
        f.write(new_code)
        
    # 3. Run the training
    print(f"⏳ Running training for Seed {seed}...")
    result = subprocess.run([sys.executable, TEMP_SCRIPT], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Training failed for Seed {seed}! Stopping pipeline.")
        sys.exit(1)
        
    # 4. FAILSAFE UPLOAD TO HUGGING FACE
    print(f"\n🚀 Uploading Seed {seed} results to Hugging Face...")
    pth_file = f"nextswinx_seed{seed}_best.pth"
    csv_file = f"training_log_seed{seed}.csv"
    
    try:
        if os.path.exists(pth_file):
            api.upload_file(
                path_or_fileobj=pth_file,
                path_in_repo=f"nextswinx_sartaj/{pth_file}", 
                repo_id=HF_REPO_ID,
                repo_type="model"
            )
            print(f"   ✅ Uploaded {pth_file}!")
            
        if os.path.exists(csv_file):
            api.upload_file(
                path_or_fileobj=csv_file,
                path_in_repo=f"nextswinx_sartaj/{csv_file}",
                repo_id=HF_REPO_ID,
                repo_type="model"
            )
            print(f"   ✅ Uploaded {csv_file}!")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        print("   ⚠️  PLEASE CHECK YOUR HUGGING FACE TOKEN. IT MUST HAVE 'WRITE' PERMISSIONS.")

print("\n" + "="*70)
print("🏆 SEEDS 43 & 44 COMPLETE!")
print("="*70)
