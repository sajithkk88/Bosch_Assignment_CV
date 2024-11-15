import random
import os
import shutil
from pathlib import Path

# Paths
weights_path = 'runs/train/bdd100k_finetune_epoch50_batch16/weights/best.pt'  # Path to trained weights
test_folder = 'data/images/test'  # Folder with test images
temp_folder = 'data/images/temp_test'  # Temporary folder to store random sample

# Ensure temporary folder is empty or create it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder)

# Select 20 random images from the test folder
image_paths = list(Path(test_folder).glob('*.*'))
selected_images = random.sample(image_paths, 20)

# Copy selected images to the temporary folder
for img_path in selected_images:
    shutil.copy(img_path, temp_folder)

# Run inference using detect.py on the selected images in the temporary folder
os.system(f"python detect.py --weights {weights_path} --source {temp_folder} --img 640 --conf 0.25 --device 0")

# Clean up temporary folder if needed (optional)
# shutil.rmtree(temp_folder)
