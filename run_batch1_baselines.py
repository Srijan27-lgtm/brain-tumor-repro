import os
import subprocess
import sys
import time
from datetime import datetime

print("="*70)
print("🚀 BATCH 1: 6 MODELS × BRISC DATASET × 3 SEEDS (18 RUNS)")
print("="*70)

MODELS = ['resnet50', 'efficientnet', 'swin', 'convnext', 'bifnn', 'twostream']
# ONLY BRISC FOR THIS BATCH
DATASETS = ["srijan2715/brisc2025-paper-pipeline"] 
SEEDS = [42, 43, 44]

log_file = "batch1_progress_log.txt"
failed_runs_file = "batch1_failed_runs_log.txt"

# Clear previous logs
open(log_file, 'w').close()
open(failed_runs_file, 'w').close()

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

start_time = time.time()
total_runs = len(MODELS) * len(DATASETS) * len(SEEDS)
current_run = 0

for model in MODELS:
    for ds in DATASETS:
        ds_name = ds.split('/')[-1]
        log(f"{'#'*70}")
        log(f"# 📂 DATASET: {ds_name.upper()} | 🤖 MODEL: {model.upper()}")
        log(f"{'#'*70}")
        
        for seed in SEEDS:
            current_run += 1
            log(f"🔥 Run {current_run}/{total_runs}: Seed {seed} for {model}...")
            
            try:
                # Run training in a separate subprocess (guarantees clean VRAM wipe between runs)
                cmd = [
                    sys.executable, "train_baselines.py",
                    "--model", model,
                    "--seed", str(seed),
                    "--dataset", ds
                ]
                
                result = subprocess.run(cmd)
                
                if result.returncode != 0:
                    raise Exception(f"Training script returned error code {result.returncode}")
                    
                log(f"✅ SUCCESS: Run {current_run}/{total_runs} complete.")
                
                # Small sleep to prevent Hugging Face API rate limits and let GPU cool slightly
                time.sleep(5) 
                
            except Exception as e:
                log(f"❌ FAILED: Run {current_run}/{total_runs} failed! Error: {e}")
                # Save to failed log so you know exactly what to re-run if needed
                with open(failed_runs_file, 'a') as f:
                    f.write(f"Failed: {model} | {ds} | Seed {seed}\n")

end_time = time.time()
total_hours = (end_time - start_time) / 3600

log(f"\n{'='*70}")
log(f"🏆 BATCH 1 FINISHED! Total Time: {total_hours:.2f} hours")
log(f"🔗 Check HF: https://huggingface.co/srijan2715/reproductioncheckpoints/tree/main/baselines")
log(f"{'='*70}")

if os.path.exists(failed_runs_file) and os.path.getsize(failed_runs_file) > 0:
    log("⚠️  WARNING: Some runs failed! Check 'batch1_failed_runs_log.txt'.")
else:
    log("🎉 ALL 18 RUNS SUCCEEDED!")
