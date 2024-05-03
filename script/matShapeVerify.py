import torch
from scipy.io import loadmat
from pathlib import Path
import os
import numpy as np
import sys
sys.path.append('')
from sparch.dataloaders.spiking_datasets import MatSpikingDataset
# Assuming you have the MatSpikingDataset class defined above this code.

def main():
    # Parameters for initializing the dataset
    data_folder = ''  # Replace with the path to your dataset folder
    split = 'train'  # or 'test'
    nb_steps = 1000  # Number of time steps (as in your class definition)

    # Initialize the dataset
    dataset = MatSpikingDataset(data_folder=data_folder, split=split, nb_steps=nb_steps)

    # Get the number of samples in the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Load the first sample
    spike_train, label = dataset[2000]

    # Print out the shapes and some stats about the spike train
    print(f"Spike train shape: {spike_train.shape}")
    print(f"Non-zero spikes in spike train: {torch.nonzero(spike_train).size(0)}")
    print(f"Label: {label}")

    # If the dataset has a method to generate batches, test it here:
    if hasattr(dataset, 'generateBatch'):
        batch = [dataset[i] for i in range(4)]  # Let's create a small batch of 4 samples
        batch_x, batch_y = dataset.generateBatch(batch)
        print(f"Batch spike train shape: {batch_x.shape}")
        print(f"Batch labels shape: {batch_y.shape}")

if __name__ == '__main__':
    main()
