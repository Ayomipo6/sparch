import numpy as np

# Replace 'your_file_path.npz' with the path to your .npz file
npz_file_path = '/Users/demiladepopoola/Development/Archive/TESS_npz/OAF_Sad/OAF_hush_sad.npz'

# Load the npz file
with np.load(npz_file_path, allow_pickle=True) as data:
    arr_0 = data['arr_0']

# Accessing the first element of the first row
first_element = arr_0[1, 0]
print(f'First element: {first_element}')

# Accessing the entire first row
first_row = arr_0[1, :]
print(f'First row: {len(first_row)}')

# Assuming the data represents two different variables/features over 2524 samples,
# you might perform operations such as computing the mean of each row
mean_of_each_row = np.mean(arr_0, axis=1)
print(f'Mean of each row: {mean_of_each_row}')