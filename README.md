<!--
SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>

SPDX-License-Identifier: BSD-3-clause

This file is part of the sparch package
--->



This [PyTorch](https://pytorch.org/) based toolkit is for developing spiking neural networks (SNNs) by training and testing them on speech command recognition tasks. It was published as part of the following paper: [A Surrogate Gradient Spiking Baseline for Speech Command Recognition](https://doi.org/10.3389/fnins.2022.865897) by A. Bittar and P. Garner (2022).


## Data

### Spiking data sets

The spiking datasets of SHD and SSC were provided by [Cramer et al. (2020)](https://doi.org/10.1109/TNNLS.2020.3044364) for speech command recognition using [LAUSCHER](https://github.com/electronicvisions/lauscher), a biologically plausible model to convert audio waveforms into spike trains based on physiological processes. However, the SSC was too large to be used for training because of the complexity it demands, theefore the original SC dataset was cut down to roughly 10% of its size (10,500 audio files) so it can be used for classification. Thereafter, it was encoded using the LAUSCHER into spiking format.  The TESS dataset was retrieved from [Toronto emotional speech set (2020)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and also encoded to spikes using the LAUSCHER dataset.



The spiking versions of the HD and SC can be downloaded from the [Zenke Lab website](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

### Non-spiking data sets

The original, SC and HD datasets are also available and can be used in this framework.Acoustic features from the dataset are extracted from the waveform usiing frame based encoding (MFCC and Mel spectrogram) and passed into the neural networks. The features are converted to spike trains inside the first hidden layer of the network. Furthermore, no prior audio processing is necessary because the first (non-trainable) transformation of the audio waveforms into features can be completed quickly enough to be completed during training.


- The Heidelberg Digits (HD) data set can also be downloaded from the same [website](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) as its spiking counterpart.

- The  Speech Command (SC) data set, introduced by [Warden (2018)](https://arxiv.org/abs/1804.03209), can be found on the [TensorFlow website](https://www.tensorflow.org/datasets/catalog/speech_commands).

The data loader implementation encompasses three distinct dataset classes, tailored to handle varied spiking dataset formats: `SpikingDataset`, `NPZSpikingDataset`, and `MatSpikingDataset`. These classes are  designed to streamline the loading and preprocessing of spiking neural network data for ASR tasks, ensuring robustness and efficiency. 

The `SpikingDataset` class is adept at managing datasets structured as HDF5 files, exemplified by the lauscher encoded HD. It efficiently parses data from HDF5 files, generating spike trains characterized by predefined time steps and unit counts. Each data sample comprises spike trains represented as sparse tensors, enabling seamless integration with neural network architectures. Additionally, this class offers methods for individual data sample retrieval and batch generation, facilitating model training and evaluation. 

 
In contrast, the `NPZSpikingDataset` class caters to spiking datasets organized as folders, where each folder corresponds to a distinct label and contains NumPy `.npz` files. Leveraging this structure, the class adeptly loads data from `.npz` files, constructs spike trains conforming to specified time step configurations, and extracts labels from folder names. Data samples are represented as dense tensors, ensuring compatibility with a wide array of machine learning frameworks. 

 
Meanwhile, the `MatSpikingDataset` class specializes in managing datasets structured as folders containing MATLAB `.mat` files, with each file constituting a single data sample. By parsing data from these files, the class generates spike trains conforming to the prescribed time step parameters. Subsequently, labels are extracted from folder names, enabling seamless association with corresponding data samples. Encapsulating data retrieval and batch generation functionalities, this class facilitates efficient utilization of spiking neural network data in model training pipelines. 



Moreover, a meticulously crafted function named `load_spiking_datasets` orchestrates the selection of the appropriate dataset class based on dataset specifications, such as name and split configuration. By initializing the chosen dataset class with user-defined parameters, this function ensures seamless integration of data loading mechanisms into machine learning workflows. Subsequently, it returns a DataLoader object, meticulously configured to facilitate efficient batch processing, encompassing options for data shuffling and parallelized loading for expedited model training and evaluation. 

## Installation

    git clone [https://github.com/idiap/sparch.git](https://github.com/Ayomipo6/sparch.git)
    cd sparch
    pip install -r requirements.txt
    python setup.py install

### Run experiments

All experiments on the speech command recognition datasets can be run from the `run_exp.py` script. The experiment configuration can be specified using parser arguments. Run the command `python run_exp.py -h` to get the descriptions of all possible options. For instance, if you want to run a new SNN experiment with adLIF neurons on the SC dataset,

    python run_exp.py --model_type adLIF --dataset_name sc \
        --data_folder <PATH-TO-DATASET-FOLDER> --new_exp_folder <OUTPUT-PATH>

You can also continue training from a checkpoint

    python run_exp.py --use_pretrained_model 1 --load_exp_folder <OUTPUT-PATH> \
        --dataset_name sc --data_folder <PATH-TO-DATASET-FOLDER> \
        --start_epoch <LAST-EPOCH-OF-PREVIOUS-TRAINING>


## Usage

Spiking neural networks (SNNs) based on the surrogate gradient approach are defined in `sparch/models/snn_models.py` as PyTorch modules. We distinguish between four types of spiking neuron models based on the linear Leaky Integrate and Fire (LIF),

- LIF: LIF neurons without layer-wise recurrent connections
- RLIF: LIF neurons with layer-wise recurrent connections
- adLIF: adaptive LIF neurons without layer-wise recurrent connections
- RadLIF: adaptive LIF neurons with layer-wise recurrent connections.

An SNN can then be simply implemented as a PyTorch module:

    from sparch.models.snns import SNN

    # Build input
    batch_size = 4
    nb_steps = 100
    nb_inputs = 20
    x = torch.Tensor(batch_size, nb_steps, nb_inputs)
    nn.init.uniform_(x)

    # Define model
    model = SNN(
        input_shape=(batch_size, nb_steps, nb_inputs),
        neuron_type="adLIF",
        layer_sizes=[128, 128, 10],
        normalization="batchnorm",
        dropout=0.1,
        bidirectional=False,
        use_readout_layer=False,
        )

    # Pass input through SNN
    y, firing_rates = model(x)


and used for other tasks. Note that by default, the last layer of the SNN is a readout layer that produces non-sequential outputs. For sequential outputs, simply set `use_readout_layer=False`. Moreover, the inputs do not have to be binary spike trains.

Standard artificial neural networks (ANNs) with non-spiking neurons are also defined in `sparch/models/ann_models.py`, in order to have a point of comparison for the spiking baseline. We implemented the following types of models: MLPs, RNNs (https://doi.org/10.1109/TETCI.2017.2762739) and [GRUs](https://doi.org/10.48550/arXiv.1406.1078).

## Structure of the git repository

```
.
├── script
|   ├── class.py
|   ├── load_mat.py
|   ├──  load_npz.py
|   ├──  matShapeVerify.py
|   ├──  max_time.py
|   ├──  prepare_datasets.py
|   ├── load_mat.py
|   ├──  split_datasets.py
    ├── validation_split_datasets.py
├── sparch
│   ├── dataloaders
|   |   ├── nonspiking_datasets.py
│   │   └── spiking_datasets.py
│   ├── models
|   |   ├── anns.py
│   │   └── snns.py
│   ├── parsers
|   |   ├── model_config.py
│   │   └── training_config.py
│   └── exp.py
|
└── run_exp.py
