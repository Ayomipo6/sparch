import os
import shutil
import random

# Set the path to your dataset directory
dataset_path = '/Users/popoola/Downloads/speech_commands_v0.02-003/'
# Set the path for the new sampled dataset directory
sampled_dataset_path = '/Users/popoola/Downloads/sampled_speech_commands/'

# Ensure the destination directory exists
os.makedirs(sampled_dataset_path, exist_ok=True)

# Set a fixed seed for reproducibility
random.seed(42)

# Iterate over each class directory in the dataset
for class_folder in os.listdir(dataset_path):
    class_dir_path = os.path.join(dataset_path, class_folder)
    
    # Check if it's a directory
    if os.path.isdir(class_dir_path):
        # List all WAV files in this class directory
        files = [file for file in os.listdir(class_dir_path) if file.endswith('.wav')]
        # Randomly select 300 files
        sampled_files = random.sample(files, 300)
        
        # Create a corresponding folder in the new sampled dataset directory
        sampled_class_dir_path = os.path.join(sampled_dataset_path, class_folder)
        os.makedirs(sampled_class_dir_path, exist_ok=True)
        
        # Copy the sampled files to the new directory
        for file in sampled_files:
            src_file_path = os.path.join(class_dir_path, file)
            dst_file_path = os.path.join(sampled_class_dir_path, file)
            shutil.copy2(src_file_path, dst_file_path)

print("Sampling complete. Check the folder:", sampled_dataset_path)
