# DL-MT-floodplain-surrogate
A discharge-driven low-rank surrogate model for predicting two-dimensional floodplain hydrodynamics

> Training dataset: The training dataset used in this study is available on [Zenodo][10.5281/zenodo.22831693]

## Overview

This repository contains the implementation of a deep-learning multi-task (DL-MT) surrogate framework for reconstructing time-resolved floodplain water-depth and velocity fields.

The framework uses the upstream discharge hydrograph as the online input. Variable-specific empirical orthogonal function (EOF) bases provide compact spatial representations, while a shared temporal network predicts the corresponding modal coefficients. The model jointly reconstructs water depth and two horizontal velocity components.

The hydraulic reference data were generated using the two-dimensional hydrodynamic model HEC-RAS version 6.6.[https://www.hec.usace.army.mil/software/hec-ras/download.aspx]

## Main features

- Builds low-rank spatial bases using Incremental PCA or truncated SVD.
- Trains multi-task models for water depth and velocity components.
- Supports ResNet and temporal convolutional network (TCN) backbones.
- Reconstructs complete flood events from upstream discharge hydrographs.
- Evaluates water-depth, velocity, and inundation-extent predictions.
- Produces spatial maps and time-step-wise evaluation results.

## Repository structure

```text
DL-MT-floodplain-surrogate/
├── flood_main.py        # Main command-line entry point
└── function/
    ├── basis.py         # Incremental-PCA basis construction
    ├── basis_builder.py # Truncated-SVD basis construction
    ├── train.py         # Model training and validation
    ├── infer.py         # Batch inference and field reconstruction
    ├── model.py         # ResNet and TCN model definitions
    ├── data.py          # Data loading utilities
    ├── losses.py        # Multi-task loss functions
    ├── metrics.py       # Evaluation metrics
    └── postviz.py       # Visualisation utilities

## Requirements

The code was developed in Python 3. Main dependencies include:
- PyTorch
- NumPy
- pandas
- h5py
- scikit-learn
- matplotlib
- tqdm
Data format
The framework uses upstream discharge hydrographs as inputs and HEC-RAS hydraulic outputs as reference fields.
- Discharge hydrographs are provided as Excel files (.xlsx).
- Hydraulic fields are obtained from HEC-RAS output files.
- Water depth and the two horizontal velocity components are represented using variable-specific low-rank bases.
Please provide data paths explicitly through the command-line arguments. The default paths in the source code are local development paths and should be replaced with paths on your system.

## Usage

View available commands:
python flood_main.py --help
Typical workflow:
# Build a low-rank basis
python flood_main.py build-basis-svd --hdf_dir <HDF_DIRECTORY> --save <BASIS_DIRECTORY>

## Train a surrogate model
python flood_main.py train --inflow <TRAIN_INFLOW_DIRECTORY> --HDF5 <TRAIN_HDF_DIRECTORY> --val_inflow <VALIDATION_INFLOW_DIRECTORY> --val_hdf5 <VALIDATION_HDF_DIRECTORY> --backbone tcn

## Reconstruct flood events
python flood_main.py infer --inflow <TEST_INFLOW_DIRECTORY> --checkpoint <MODEL_CHECKPOINT> --norm_stats <NORMALISATION_FILE> --output_dir <OUTPUT_DIRECTORY>

##  Visualise predictions and calculate metrics
python flood_main.py viz --pred_dir <PREDICTION_DIRECTORY> --gt_dir <REFERENCE_DIRECTORY> --output_dir <FIGURE_DIRECTORY>
Run python flood_main.py <command> --help to view complete arguments for each command.

##  Citation
If you use this code or dataset, please cite:
Qiu, Y., et al. A discharge-driven low-rank surrogate model for predicting two-dimensional floodplain hydrodynamics. Environmental Modelling & Software (under review).

## License
This project is distributed under the MIT License. See the LICENSE file for details.
