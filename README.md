# DL-MT Floodplain Surrogate

Code for **A discharge-driven low-rank framework for joint prediction of flood depth and velocity to support rapid flood warning**.

DL-MT reconstructs time-varying water-depth and velocity fields from upstream discharge hydrographs for a fixed domain and prescribed hydraulic conditions. Variable-specific EOF bases provide low-rank spatial representations, and a shared Res-MT or TCN-MT network predicts modal coefficients for field reconstruction.

- Data: [Zenodo](https://doi.org/10.5281/zenodo.22831693)
- Licence: [MIT](LICENSE)

## Installation

Use Python 3.10 or later. Install PyTorch for your CPU or CUDA environment, then install the remaining dependencies:

```bash
pip install torch numpy pandas openpyxl h5py scikit-learn scipy matplotlib tqdm rasterio pyproj
```

Run commands from the repository directory containing `flood_main.py`. CPU inference is supported; a CUDA GPU can be used for training. The manuscript experiments used an NVIDIA RTX 3090 with 24 GB memory for training and an Intel Xeon Platinum 8168 CPU for inference.

## Inputs and outputs

- **Inputs:** one header-free Excel file per event, containing the prepared input features with 24 rows and 26 columns. Use numeric event filenames, such as `01.xlsx`.
- **Reference fields:** matching HEC-RAS HDF files, such as `01.hdf`, containing depth and the two horizontal velocity components. Dataset paths are configured in `function/paths.py`.
- **Model files:** trained checkpoint, variable-specific EOF bases and mean fields, and training-set normalisation statistics.
- **Outputs:** reconstructed depth and velocity fields in NPY format, with evaluation and plotting utilities.

Use separate training, validation and test events. Construct EOF bases and normalisation statistics from training events only. Replace the uppercase placeholders below with your own file or directory paths.

## Basic workflow

### 1. Build EOF bases

Run this command separately for depth and both velocity components, using each variable's HDF dataset path and output directory:

```bash
python flood_main.py build-basis-svd --hdf_dir TRAIN_HDF --data_path "HDF_DATASET_PATH" --rank_k 256 --save BASIS_DIRECTORY
```

### 2. Train

```bash
python flood_main.py train --inflow TRAIN_INPUT --HDF5 TRAIN_HDF --val_inflow VAL_INPUT --val_hdf5 VAL_HDF --depth_basis DEPTH_BASIS --vx_basis VX_BASIS --vy_basis VY_BASIS --backbone tcn --tasks depth,flow --save_dir RUNS
```

Training saves the configuration, checkpoint and normalisation statistics. Backbone options are listed by `python flood_main.py train --help`.

### 3. Reconstruct hydraulic fields

```bash
python flood_main.py infer --inflow TEST_INPUT --checkpoint CHECKPOINT --norm_stats TRAIN_NORM_NPZ --depth_basis DEPTH_BASIS --vx_basis VX_BASIS --vy_basis VY_BASIS --mean_depth DEPTH_MEAN --mean_vx VX_MEAN --mean_vy VY_MEAN --backbone tcn --device cpu --output_dir PREDICTIONS
```

Use the backbone, bases and normalisation statistics associated with the checkpoint.

### 4. Evaluate and visualise

```bash
python flood_main.py viz --pred_dir PREDICTIONS --gt_dir TEST_HDF --geom_hdf GEOMETRY_HDF --threshold 0.1 --output_dir FIGURES
```

Use `python flood_main.py COMMAND --help` to view the options for each command.

## Parameters

| Parameter | Code default |
|---|---|
| EOF rank | 256 per variable |
| Initial learning rate | 0.001 |
| Batch size | 4 |
| Maximum epochs / early-stopping patience | 500 / 30 |
| Wet/dry depth threshold | 0.1 m |
| TCN channels / dilation factors | 64, 64, 64 / 1, 2, 4 |
| TCN kernel size / dropout | 3 / 0.2 |
| Depth-weighted loss coefficient / exponent | 0.5 / 1.0 |
| Huber delta / weight-clipping quantile | 0.1 m / 0.999 |
| Dry-region penalty coefficient | 0.5 |

The manuscript examines dry-region penalty coefficients of **0, 0.25, 0.5, 0.75, 1 and 2**, and training-set sizes of **20, 40, 60, 80 and 100 events**, with fixed validation and test sets of 10 events each.

## Citation

Qiu, Y., et al. *A discharge-driven low-rank framework for joint prediction of flood depth and velocity to support rapid flood warning*. Manuscript. Please also cite the dataset DOI when using the data.
