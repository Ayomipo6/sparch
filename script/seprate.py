
import os
import scipy.io
from shutil import copy2

# Paths to the directories containing the encoded and original files
encoded_dir =  '/Users/popoola/Development/Archive/encoded_signals'
original_dir = '/Users/popoola/Downloads/speech_commands_v0.02-003/'

output_dir = '/Users/popoola/Downloads/sampled/speech_commands_v0.02-003'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each class folder in the encoded directory
for class_folder in os.listdir(encoded_dir):
    class_path = os.path.join(encoded_dir, class_folder)
    
    if os.path.isdir(class_path):  # Check if it is a directory
        # Iterate over each .mat file in the class directory
        for mat_file in os.listdir(class_path):
            if mat_file.endswith('.mat'):
                # Load the .mat file (optional, if needed to process something inside)
                mat_path = os.path.join(class_path, mat_file)
                mat_data = scipy.io.loadmat(mat_path)
                
                # Corresponding .wav file path
                wav_path = os.path.join(original_dir, class_folder, mat_file.replace('.mat', '.wav'))
                
                if os.path.exists(wav_path):
                    # Directory in the output for this class
                    output_class_dir = os.path.join(output_dir, class_folder)
                    os.makedirs(output_class_dir, exist_ok=True)
                    
                    # Copy the .wav file to the output class directory
                    copy2(wav_path, output_class_dir)

print("Data organization complete.")
