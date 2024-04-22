#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the dataloader is defined for the SHD and SSC datasets.
"""
import logging
from pathlib import Path
import os
import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file or npz).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        dataset_name,
        data_folder,
        split,
        nb_steps=100,
    ):

        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=np.int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)
        y = self.labels[index]

        return x.to_dense(), y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        return xs, xlens, ys

class NPZSpikingDataset(Dataset):
    """
    Dataset class for a spiking dataset, where data is organized in folders,
    each named by the label, and containing .npz files.

    Arguments
    ---------
    data_folder : str
        Path to the root folder containing the dataset folders.
    split : str
        Split of the dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        data_folder,
        split,
        nb_steps=100,
    ):
        # Fixed parameters
        self.device = "cpu"
        self.nb_steps = nb_steps
        self.nb_units = 700  # Adjust as needed
        self.max_time = 2.9
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder

        def load_list(filename):
            filepath = os.path.join(self.data_folder, filename)
            with open(filepath) as f:
                return [os.path.join(self.data_folder, i.strip()) for i in f]

        if split == "train":
            files = sorted(str(p) for p in Path(data_folder).glob("*/*.npz"))
            exclude = load_list("test_list.txt")
            exclude = set(exclude)
            self.file_list = [
                w for w in files if w not in exclude
            ]
        else:
            self.file_list = load_list(str(split) + "_list.txt")
        
        self.labels = sorted(next(os.walk(data_folder))[1])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # Read waveform
        filename = self.file_list[index]
        with np.load(filename, allow_pickle=True) as data:
            spikes = data['arr_0']
        firing_times = spikes[0, :]
        units_fired = spikes[1, :]

        times = np.digitize(firing_times, self.time_bins) - 1  # Adjust indices to be 0-based
        units = units_fired.astype(int)

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)

        # Extract label from the file path
        label_name = Path(filename).parent.name
        label_index = self.labels.index(label_name)  # Convert folder name to index
        y = torch.tensor(label_index, dtype=torch.long).to(self.device)

        return x.to_dense(), y  # Returning dense tensor for compatibility with many models

    def generateBatch(self, batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        return xs, xlens, ys

class MatSpikingDataset(Dataset):
    """
    Dataset class for a spiking dataset, where data is organized in folders,
    each named by the label, and containing .mat files.

    Arguments
    ---------
    data_folder : str
        Path to the root folder containing the dataset folders.
    split : str
        Split of the dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        data_folder,
        split,
        nb_steps=100,
    ):
        self.max_time = 2.40
        self.device = "cpu"
        self.nb_steps = nb_steps
        self.nb_units = 120  # This is based on the 'C' value range from your provided structure.
        self.data_folder = data_folder

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

        # Load list of .mat files
        def load_list(filename):
            filepath = os.path.join(self.data_folder, filename)
            with open(filepath) as f:
                return [os.path.join(self.data_folder, i.strip()) for i in f]

        if split == "train":
            files = sorted(str(p) for p in Path(data_folder).glob("*/*.mat"))
            exclude = load_list("test_list.txt")
            exclude = set(exclude)
            self.file_list = [
                w for w in files if w not in exclude
            ]
        else:
            self.file_list = load_list(str(split) + "_list.txt")
        
        self.labels = sorted(next(os.walk(data_folder))[1])

    def __len__(self):
        return len(self.file_list)

    # def __getitem__(self, index):
        # Load .mat file
        filename = self.file_list[index]
        mat_data = loadmat(filename)

        # Extract the spiking data
        spikes = mat_data['soft_spikegrams']
        firing_times = spikes[:, 2]  # T values
        firing_channels = spikes[:, 0].astype(int) - 1  # C values, converting to 0-based index

        # Normalize the time values to fit into the number of steps
        time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
        times = np.digitize(firing_times, time_bins) - 1

        # Create the input tensor
        x = torch.zeros(self.nb_steps, self.nb_units)
        x[times, firing_channels] = torch.from_numpy(spikes[:, 1]).float()  # Convert to Float

        # Extract label from the file path
        label_name = Path(filename).parent.name
        label_index = self.labels.index(label_name)  # Convert folder name to index
        y = torch.tensor(label_index, dtype=torch.long)

        return x, y
   
    def __getitem__(self, index):
        # Load .mat file
        filename = self.file_list[index]
        mat_data = loadmat(filename)

        # Extract the spiking data
        spikes = mat_data['soft_spikegrams']
        firing_times = spikes[:, 2]  # T values
        firing_channels = spikes[:, 0].astype(int) - 1  # C values, converting to 0-based index

        # Dynamically determine max_time for this specific data item
        item_max_time = firing_times.max()

        # Normalize the time values to fit into the number of steps
        # Using item-specific max_time
        time_bins = np.linspace(0, item_max_time, num=self.nb_steps)
        times = np.digitize(firing_times, time_bins) - 1

        # Ensure times are within bounds after digitization
        times = np.clip(times, 0, self.nb_steps - 1)

        # Create the input tensor
        x = torch.zeros(self.nb_steps, self.nb_units, dtype=torch.float)
        for t, c in zip(times, firing_channels):
            x[t, c] += 1  # Increment to account for possible multiple spikes in the same bin for a channel

        # Extract label from the file path
        label_name = Path(filename).parent.name
        label_index = self.labels.index(label_name)  # Convert folder name to index
        y = torch.tensor(label_index, dtype=torch.long)

        return x, y

    def generateBatch(self, batch):
            # Batch generation method adapted for .mat data
            xs, ys = zip(*batch)
            xs = torch.stack(xs)
            ys = torch.stack(ys)

            return xs, ys

def load_spiking_datasets(
    dataset_name,
    data_folder,
    split,
    batch_size,
    nb_steps=100,
    shuffle=True,
    workers=0,
):
    """
    This function creates a dataloader for a given split of
    the SHD or SSC datasets.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of dataset, must be either "train" or "test" for SHD.
        For SSC, can be "train", "valid" or "test".
    batch_size : int
        Number of examples in a single generated batch.
    shuffle : bool
        Whether to shuffle examples or not.
    workers : int
        Number of workers.
    """
    if dataset_name not in ["shd", "lautess", "ssc", "spitess", "spihd"]:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")

    if dataset_name == "shd" and split == "valid":
        logging.info("SHD does not have a validation split. Using test split.")
        split = "test"
    if dataset_name == "lautess" and split == "valid":
        logging.info("TESS does not have a validation split. Using test split.")
        split = "test"
    if dataset_name == "spitess" and split == "valid":
        logging.info("TESS does not have a validation split. Using test split.")
        split = "test"
    if dataset_name == "spihd" and split == "valid":
        logging.info("TESS does not have a validation split. Using test split.")
        split = "test"

    if dataset_name == "lautess":
        dataset = NPZSpikingDataset(data_folder, split, nb_steps)
    elif dataset_name == "spitess":
        dataset = MatSpikingDataset(data_folder, split, 100)
    elif dataset_name == "spihd":
        dataset = MatSpikingDataset(data_folder, split, 1000)
    else:
        dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps)
        
    logging.info(f"Number of examples in {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader
