# DL-MT Floodplain Surrogate

A discharge-driven low-rank framework for joint prediction of flood depth and velocity to support rapid flood warning.

The framework converts upstream discharge hydrographs and their derived features into time-resolved depth and horizontal velocity fields for a fixed domain and prescribed hydraulic conditions. HEC-RAS 6.6 supplies the hydraulic reference data. Variable-specific EOF bases are constructed offline; a shared ResNet or TCN predicts modal coefficients for field reconstruction.

Dataset: [Zenodo](https://doi.org/10.5281/zenodo.22831693)

## Requirements

Python 3.10 or later, PyTorch, NumPy, pandas, openpyxl, h5py, scikit-learn, matplotlib, scipy, tqdm, rasterio, and pyproj. CUDA is optional for training; inference defaults to CPU.

## Workflow

Run commands from the directory containing `flood_main.py`. Replace all example paths with your own; source-code defaults include local development paths.

1. **Prepare data.** Use separate event sets for training, validation, and testing. Inputs are Excel feature matrices (24 time steps by 26 features), with filenames matching the hydraulic reference event IDs. HDF dataset paths are defined in `function/paths.py`.
2. **Build bases from training events only.** Repeat for depth, velocity X, and velocity Y using the corresponding HEC-RAS dataset path.

```bash
python flood_main.py build-basis-svd --hdf_dir TRAIN_HDF --data_path "HDF_DATASET_PATH" --rank_k 256 --save BASIS_DIRECTORY
```

3. **Train and validate.** Supply the three variable-specific bases. Each run saves its configuration, checkpoint, and training normalisation statistics.

```bash
python flood_main.py train --inflow TRAIN_INPUT --HDF5 TRAIN_HDF --val_inflow VAL_INPUT --val_hdf5 VAL_HDF --depth_basis DEPTH_BASIS --vx_basis VX_BASIS --vy_basis VY_BASIS --backbone tcn --tasks depth,flow --save_dir RUNS
```

4. **Reconstruct independent events on CPU.** Match the backbone and basis settings to the checkpoint. Use `--device cuda` or `--device auto` only when a different execution device is intended.

```bash
python flood_main.py infer --inflow TEST_INPUT --checkpoint CHECKPOINT --norm_stats TRAIN_NORM_NPZ --depth_basis DEPTH_BASIS --vx_basis VX_BASIS --vy_basis VY_BASIS --mean_depth DEPTH_MEAN --mean_vx VX_MEAN --mean_vy VY_MEAN --backbone tcn --device cpu --output_dir PREDICTIONS
```

5. **Evaluate and visualise.** Depth, velocity-component, and speed predictions are saved as NPY arrays. Provide the matching HEC-RAS geometry and reference data for maps and metrics.

```bash
python flood_main.py viz --pred_dir PREDICTIONS --gt_dir TEST_HDF --geom_hdf GEOMETRY_HDF --threshold 0.1 --output_dir FIGURES
```

Use `python flood_main.py COMMAND --help` for all options. Optional analysis and preprocessing scripts are in `assit_code/`.

## Configuration

- Wet/dry depth threshold: 0.1 m. The event-envelope speed classification threshold is 0.1 m/s; plotting visibility thresholds are separate.
- Default TCN: three temporal blocks with 64 channels each and dilation factors 1, 2, and 4; each block contains two dilated convolutions.
- Restored depth-weighted-loss defaults: alpha = 0.5, gamma = 1.0, Huber delta = 0.1 m, clipping quantile = 0.999, and no warmup. The dry-region coefficient is 0.5.
- The restored loss defaults are a configuration for validation, not a claim that existing checkpoints or manuscript metrics were produced with these settings. Use recorded experiment configurations to reproduce published results.
- CPU inference is available through `--device cpu` or `function.infer.batch_infer_cpu(args)`. Report timing together with the execution device and measurement scope; the current event timer includes input reading, reconstruction, and output saving.

## Citation and licence

For now, cite the associated manuscript as an unpublished manuscript and cite the dataset DOI above. Update this entry with the article DOI after publication.

Code licence: MIT; see the repository's `LICENSE` file. The dataset licence is specified in its Zenodo record.
