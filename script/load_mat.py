import numpy as np
from scipy.io import loadmat

# Path to your .mat file
mat_file_path = ''

# Load the .mat file
mat_data = loadmat(mat_file_path)

# Assuming your data is stored under the key 'soft_spikegrams'
# Replace 'soft_spikegrams' with the actual key if different
data = mat_data['soft_spikegrams']

# Accessing the first element of the first row
first_element = data[1, 0]
print(f'First element: {first_element}')

# Accessing the entire first row
first_row = data[1, :]
print(f'First row length: {len(first_row)}')

# Assuming the data represents spikes over samples,
# you might perform operations such as computing the mean of each row
mean_of_each_row = np.mean(data, axis=1)
print(f'Mean of each row: {mean_of_each_row}')

# Extract the Intensity (I) values from the first five spikes
intensity_first_five_spikes = data[:5, 2]  # Assuming 0: Channel, 1: Intensity, 2: Time, 3: Kernel Index

print("Intensity of the first five spikes:", intensity_first_five_spikes)
