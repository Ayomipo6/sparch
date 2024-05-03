import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_dir, training_list_path, validation_list_path, testing_list_path):
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

    # Split the DataFrame into training + validation set and test set (80/20 split)
    train_val_df, test_df = train_test_split(df_files, test_size=0.2, random_state=40, stratify=df_files['label'])

    # Further split the training + validation set into actual training and validation sets (87.5/12.5 split)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=40, stratify=train_val_df['label'])

    # Write the training, validation, and testing file paths to their respective text files
    train_df['file_path'].to_csv(training_list_path, header=False, index=False)
    val_df['file_path'].to_csv(validation_list_path, header=False, index=False)
    test_df['file_path'].to_csv(testing_list_path, header=False, index=False)

# Example usage
dataset_dir = '/Users/ayo/datasets/ssc'
training_list_path = '/Users/ayo/datasets/ssc/training_list.txt'
validation_list_path = '/Users/ayo/datasets/ssc/validation_list.txt'
testing_list_path = '/Users/ayo/datasets/ssc/testing_list.txt'

prepare_dataset(dataset_dir, training_list_path, validation_list_path, testing_list_path)
