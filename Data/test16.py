import os
import shutil
from tqdm import tqdm

# Define source and destination directories
source_dir = "/home/vipuser/Desktop/nnUNet/nnUNet_output/PASp61_3d_predict_labelsTr"
dest_dir = "/home/vipuser/Desktop/Data/Task02_PASp61/labelsTr"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Get all .nii.gz files from the source directory
nii_files = [f for f in os.listdir(source_dir) if f.endswith(".nii.gz")]
print(f"Found {len(nii_files)} .nii.gz files in source directory")

# Copy files to destination with a progress bar
for file in tqdm(nii_files, desc="Copying .nii.gz files"):
    source_path = os.path.join(source_dir, file)
    dest_path = os.path.join(dest_dir, file)
    shutil.copy2(source_path, dest_path)

print(f"Successfully copied {len(nii_files)} .nii.gz files to {dest_dir}")