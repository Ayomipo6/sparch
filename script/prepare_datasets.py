import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_dir, training_list_path, testing_list_path):
    # List all files in the dataset directory
    all_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.npz'):
                relative_path = os.path.relpath(root, dataset_dir)
                all_files.append((os.path.join(relative_path, file), os.path.basename(relative_path)))
    
    # Create a DataFrame from the file paths and labels
    df_files = pd.DataFrame(all_files, columns=['file_path', 'label'])

    # Trim any leading/trailing whitespace from the file paths (if necessary)
    df_files['file_path'] = df_files['file_path'].str.strip()

    # Split the DataFrame into training and testing sets (80/20 split) with stratification
    train_df, test_df = train_test_split(df_files, test_size=0.2, random_state=40, stratify=df_files['label'])

    # Write the training and testing file paths to their respective text files
    train_df['file_path'].to_csv(training_list_path, header=False, index=False)
    test_df['file_path'].to_csv(testing_list_path, header=False, index=False)

# Example usage
dataset_dir = '/Users/demiladepopoola/Development/Archive/TESS_npz'  # Update this to the path where your dataset is extracted
training_list_path = '/Users/demiladepopoola/Development/Archive/TESS_npz/train_list.txt'
testing_list_path = '/Users/demiladepopoola/Development/Archive/TESS_npz/test_list.txt'

prepare_dataset(dataset_dir, training_list_path, testing_list_path)
