import os
import subprocess
import sys
from huggingface_hub import HfApi

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEEDS = [42, 43, 44]
DATASET_NAME = "srijan2715/sartaj-final-split"  # <-- CHANGE THIS IF YOUR SARTAJ REPO HAS A DIFFERENT NAME
BASE_SCRIPT = "train_pretrained.py"
TEMP_SCRIPT = "train_sartaj_temp.py"
HF_REPO_ID = "srijan2715/reproductioncheckpoints"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

print("="*70)
print(f"🚀 STARTING SARTAJ PIPELINE FOR SEEDS {SEEDS}")
print(f"📂 Dataset: {DATASET_NAME}")
print("="*70)

# 1. Read the original script
with open(BASE_SCRIPT, 'r') as f:
    original_code = f.read()

api = HfApi(token=HF_TOKEN)

# 2. Loop through seeds
for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"🔥 PREPARING RUN FOR SEED {seed}...")
    print(f"{'='*70}")
    
    # Modify the code dynamically
    new_code = original_code.replace('SEED = 42', f'SEED = {seed}')
    new_code = new_code.replace('SEED = 43', f'SEED = {seed}') # Fallback if you changed it manually
    new_code = new_code.replace('SEED = 44', f'SEED = {seed}') # Fallback
    
    # Update dataset name (Targeting the exact string from your screenshot)
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
                path_in_repo=f"nextswinx_sartaj/{pth_file}", # Organized in a separate folder
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

print("\n" + "="*70)
print("🏆 ALL SARTAJ SEEDS COMPLETE!")
print("="*70)
