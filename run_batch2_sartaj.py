import os
import subprocess
import sys
import time
from datetime import datetime

print("="*70)
print("🚀 BATCH 2: 7 MODELS × SARTAJ DATASET × 3 SEEDS (21 RUNS)")
print("="*70)

MODELS = ['resnet50', 'efficientnet', 'swin', 'convnext', 'bifnn', 'twostream', 'nextswinx']
DATASETS = ["srijan2715/sartaj-final-split"] 
SEEDS = [42, 43, 44]

log_file = "batch2_progress_log.txt"
failed_runs_file = "batch2_failed_runs_log.txt"

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
                # All models now use the unified train_baselines.py script!
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
                time.sleep(5) # Prevent HF rate limits
                
            except Exception as e:
                log(f"❌ FAILED: Run {current_run}/{total_runs} failed! Error: {e}")
                with open(failed_runs_file, 'a') as f:
                    f.write(f"Failed: {model} | {ds} | Seed {seed}\n")

end_time = time.time()
total_hours = (end_time - start_time) / 3600

log(f"\n{'='*70}")
log(f"🏆 BATCH 2 FINISHED! Total Time: {total_hours:.2f} hours")
log(f"🔗 Check HF: https://huggingface.co/srijan2715/reproductioncheckpoints/tree/main/baselines")
log(f"{'='*70}")

if os.path.exists(failed_runs_file) and os.path.getsize(failed_runs_file) > 0:
    log("⚠️  WARNING: Some runs failed! Check 'batch2_failed_runs_log.txt'.")
else:
    log("🎉 ALL 21 RUNS SUCCEEDED!")
