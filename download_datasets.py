import os
import sys
import subprocess

print("🔽 FoodSafe-IoT Dataset Downloader\n")

# Create folders
os.makedirs('01_datasets/images/fresh_rotten_classification', exist_ok=True)

print("⬇️ Downloading: swoyam2609/fresh-and-stale-classification")
try:
    subprocess.run([
        sys.executable, '-m', 'kaggle', 'datasets', 'download',
        '-d', 'swoyam2609/fresh-and-stale-classification',
        '-p', '01_datasets/images/fresh_rotten_classification',
        '--unzip'
    ], check=True)
    print("✅ Downloaded and extracted successfully!")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n🎉 Done! Check 01_datasets/images/fresh_rotten_classification/")
