# DL-MT-floodplain-surrogate

**A discharge-driven low-rank framework for joint prediction of flood depth and velocity to support rapid flood warning**

Python implementation accompanying the manuscript by Yanbin Qiu and co-authors.

- **Source code:** [GitHub repository](https://github.com/ydkyly/DL-MT-floodplain-surrogate)
- **Dataset reference:** [Zenodo, DOI: 10.5281/zenodo.22831693](https://doi.org/10.5281/zenodo.22831693)
- **Software licence:** [MIT](LICENSE)

## Overview

DL-MT is a discharge-driven, low-rank multi-task surrogate for reconstructing time-varying floodplain water depth and horizontal velocity components. It provides a computational link between a supplied upstream discharge hydrograph and spatial hydraulic information that can support rapid flood warning.

The framework combines variable-specific empirical orthogonal function (EOF) representations with shared temporal feature extraction. Separate decoding branches predict modal coefficients for water depth and the two horizontal velocity components. These coefficients are combined with their corresponding spatial bases and mean fields to reconstruct hydraulic outputs. Training objectives are evaluated on the reconstructed fields.

For a fixed computational domain and prescribed hydraulic conditions, the upstream discharge hydrograph is the only external online forcing input. The software uses a prepared feature matrix derived from that hydrograph and its time coordinates. Trained weights, EOF bases, mean fields, and training-set normalisation statistics are also required as model assets. Online inference does not call HEC-RAS or require reference hydraulic fields.

The hydraulic reference simulations were generated using [HEC-RAS 6.6](https://www.hec.usace.army.mil/software/hec-ras/). HEC-RAS is needed to generate new reference simulations, but not to run the trained surrogate on compatible inputs.

## Main capabilities

- Construct variable-specific low-rank spatial representations using incremental PCA or truncated SVD.
- Train residual convolutional (Res-MT) and temporal convolutional (TCN-MT) models within a common multi-task framework.
- Jointly reconstruct water depth, horizontal velocity components, and velocity magnitude.
- Derive inundation extent from reconstructed water depth.
- Evaluate hydraulic-field errors and inundation-extent agreement against reference simulations.
- Generate spatial plots and time-step-wise evaluation outputs.

## Repository layout

```text
DL-MT-floodplain-surrogate/
├── LICENSE
├── README.md
└── code/
    ├── flood_main.py          # Command-line entry point
    └── function/
        ├── basis.py           # Incremental-PCA basis construction
        ├── basis_builder.py   # Truncated-SVD basis construction
        ├── basis_checker.py   # Basis reconstruction checks
        ├── train.py           # Training and validation
        ├── infer.py           # Inference and field reconstruction
        ├── model.py           # Residual and temporal convolutional networks
        ├── data.py            # Input and reference-field loading
        ├── paths.py           # HEC-RAS HDF dataset paths
        ├── losses.py          # Training objectives
        ├── metrics.py         # Evaluation metrics
        ├── postviz.py         # Spatial plots and evaluation outputs
        └── tools_dump_xy.py   # Computational-cell coordinate export
```

Additional helper modules are included in `code/function/`.

## Installation

Use Python 3.10 or later; the source uses Python 3.10 union-type syntax. The exact environment used for a numerical reproduction must match the original experiment environment. The installation below describes the workflow and is not a pinned environment specification.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/ydkyly/DL-MT-floodplain-surrogate.git
cd DL-MT-floodplain-surrogate
python -m venv .venv
```

Activate the environment using the command appropriate for your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install PyTorch using the instructions for your operating system and CPU/CUDA configuration at [PyTorch](https://pytorch.org/get-started/locally/). The implementation uses `torch.amp.autocast` and `torch.amp.GradScaler`; the installed PyTorch version must provide these APIs.

Install the remaining core packages:

```bash
python -m pip install numpy pandas openpyxl h5py scikit-learn matplotlib tqdm
```

`openpyxl` is used to read `.xlsx` inputs. GeoTIFF background plotting additionally requires `rasterio`:

```bash
python -m pip install rasterio
```

Run commands from the code directory:

```bash
cd code
python flood_main.py --help
```

## Data and model assets

### Discharge input files

Each event is represented by a header-free Excel file. The current loader expects a prepared matrix with **24 rows and 26 columns**, transposed internally to 26 channels and 24 time steps. Event filenames must have numeric stems, such as `01.xlsx`, because the loader sorts them numerically.

Use the feature organisation of the study input files. A single-column discharge series cannot be passed directly to this loader. The 26-column representation must preserve the original feature ordering and time convention; `data.py` reads and normalises these features but does not construct them from a raw hydrograph.

Normalisation statistics are fitted using training inputs and reused for validation and inference. Do not refit them on validation or test events.

### Hydraulic reference files

For training and evaluation, pair each input with a reference file bearing the same event identifier, for example `01.xlsx` and `01.hdf`. Multi-task training requires water depth and both velocity components.

The current HDF paths are configured in `code/function/paths.py`. For the study area named `Perimeter 1`, the hydraulic datasets are under:

```text
Results/Unsteady/Output/Output Blocks/Base Output/
Unsteady Time Series/2D Flow Areas/Perimeter 1/
```

The dataset names are:

- `Cell Hydraulic Depth`
- `Cell Velocity - Velocity X`
- `Cell Velocity - Velocity Y`

For another HEC-RAS model, check the HDF hierarchy and update `paths.py` and the basis-construction dataset paths consistently. Cell ordering, spatial masks, units, and time steps must agree between reference fields and spatial bases.

### Required inference assets

Inference requires a compatible model checkpoint, training-set normalisation statistics, and the variable-specific bases and mean fields used with that checkpoint. Preserve these assets together with the model configuration.

Reference HDF files are required for accuracy evaluation and geometry-based plotting, not for predicting hydraulic fields from discharge inputs.

## Example workflow

The commands below are workflow templates. Replace every angle-bracket placeholder with a local path. They are not a complete specification of the manuscript's experimental runs: architecture, loss settings, event lists, and other parameters must match the corresponding experiment records.

Many source-code defaults point to the developers' local directories. Supply your own paths explicitly.

### 1. Construct spatial bases from training events

Build separate bases for depth, x-velocity, and y-velocity. For example:

```bash
python flood_main.py build-basis-svd --hdf_dir "<TRAIN_HDF_DIRECTORY>" --save "<DEPTH_BASIS_DIRECTORY>" --rank_k 256 --data_path "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Hydraulic Depth"

python flood_main.py build-basis-svd --hdf_dir "<TRAIN_HDF_DIRECTORY>" --save "<VX_BASIS_DIRECTORY>" --rank_k 256 --data_path "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity X"

python flood_main.py build-basis-svd --hdf_dir "<TRAIN_HDF_DIRECTORY>" --save "<VY_BASIS_DIRECTORY>" --rank_k 256 --data_path "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y"
```

The default SVD dataset path is for y-velocity, so specifying `--data_path` is essential when constructing the other bases. Preserve the centred mean fields with their corresponding bases. Use only training events for basis construction and keep the bases fixed during validation and testing.

Additional options are available through:

```bash
python flood_main.py build-basis-svd --help
python flood_main.py build-basis-ipca --help
python flood_main.py check-basis --help
```

### 2. Train a surrogate

```bash
python flood_main.py train --inflow "<TRAIN_INFLOW_DIRECTORY>" --HDF5 "<TRAIN_HDF_DIRECTORY>" --val_inflow "<VALIDATION_INFLOW_DIRECTORY>" --val_hdf5 "<VALIDATION_HDF_DIRECTORY>" --depth_basis "<DEPTH_BASIS_DIRECTORY>/basis.npy" --vx_basis "<VX_BASIS_DIRECTORY>/basis.npy" --vy_basis "<VY_BASIS_DIRECTORY>/basis.npy" --backbone tcn --tasks "depth,flow" --rank_k 256 --timesteps 24 --epochs 500 --batch_size 4 --lr 0.001 --early_stop_patience 30 --flood_threshold 0.1 --save_dir "<TRAINING_OUTPUT_DIRECTORY>"
```

Available backbone names are `tcn`, `resnet_basic`, and `resnet_bottleneck`. Match `--tcn_channels`, `--tcn_kernel`, and `--tcn_dropout` to the intended model configuration.

Depth-weighted loss settings are controlled by `--alpha_depth_loss`, `--alpha_target`, `--depth_loss_gamma`, and `--gamma_target`. Their current defaults are zero. To reproduce an experiment using the depth-weighted term, supply the coefficients from that experiment rather than relying on these defaults. The dry-area penalty is controlled by `--beta_dry`.

Training saves checkpoints and normalisation statistics in the run directory, including `best_<tag>.pth` and `norm_stats_<tag>.npz`. Retain the selected checkpoint, its matching statistics, the bases, and the full training configuration.

```bash
python flood_main.py train --help
```

### 3. Reconstruct independent flood events

```bash
python flood_main.py infer --inflow "<TEST_INFLOW_DIRECTORY>" --checkpoint "<MODEL_CHECKPOINT>" --norm_stats "<NORMALISATION_FILE>" --depth_basis "<DEPTH_BASIS_DIRECTORY>/basis.npy" --vx_basis "<VX_BASIS_DIRECTORY>/basis.npy" --vy_basis "<VY_BASIS_DIRECTORY>/basis.npy" --mean_depth "<DEPTH_BASIS_DIRECTORY>/mean.npy" --mean_vx "<VX_BASIS_DIRECTORY>/mean.npy" --mean_vy "<VY_BASIS_DIRECTORY>/mean.npy" --backbone tcn --tasks "depth,flow" --timesteps 24 --flood_threshold 0.1 --output_dir "<PREDICTION_DIRECTORY>"
```

Use the same architecture settings as the checkpoint, including any non-default TCN channel configuration. Do not substitute bases or normalisation statistics from another run.

For a multi-task model, the saved outputs include:

| File | Contents |
| --- | --- |
| `<event>_Y_pred.npy` | Reconstructed water depth |
| `<event>_Vx_pred.npy` | Horizontal x-velocity component |
| `<event>_Vy_pred.npy` | Horizontal y-velocity component |
| `<event>_Vmag_pred.npy` | Velocity magnitude |

Outputs are arranged by time step and computational cell. The current inference implementation clamps negative predicted depths to zero and sets velocity to zero in cells classified as dry using the selected depth threshold. Account for these output-processing steps when comparing results.

### 4. Visualise predictions and calculate metrics

```bash
python flood_main.py viz --pred_dir "<PREDICTION_DIRECTORY>" --gt_dir "<TEST_HDF_DIRECTORY>" --geom_hdf "<REFERENCE_HDF_WITH_GEOMETRY>" --area_name "Perimeter 1" --threshold 0.1 --bg_image "" --output_dir "<FIGURE_DIRECTORY>"
```

The example disables the optional background image. If adding a background, supply its actual path and ensure its coordinate system agrees with the computational geometry. Check the coordinate-conversion settings for your data.

```bash
python flood_main.py viz --help
```

## Hardware and runtime

The current training and inference functions automatically select CUDA when `torch.cuda.is_available()` returns `True`; otherwise they use the CPU. There is currently no `--device` option in `flood_main.py`.

To request CPU inference on a CUDA-enabled computer, hide CUDA devices before starting a fresh Python process. For Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

For a POSIX shell:

```bash
export CUDA_VISIBLE_DEVICES=-1
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Confirm that the check reports `False`, then run the inference command in the same shell. These settings apply to the current shell and its child processes.

The manuscript reports CUDA-accelerated training on an NVIDIA RTX 3090 with 24 GB memory, and a runtime comparison using an Intel Xeon Platinum 8168 CPU for surrogate inference and HEC-RAS. These are reported experimental conditions, not a statement that the current script defaults reproduce that benchmark. Record the actual device, software versions, thread settings, timed operations, and repeat procedure when measuring runtime.

Memory demand depends on the number of events, time steps, retained cells, basis rank, and reconstruction-block size. The reported hardware is not a minimum-system specification.

## Reproducing the study

The study uses 120 synthetic flood scenarios split by event into 100 training, 10 validation, and 10 test events. All snapshots from one event belong to the same subset. EOF bases, mean fields, and input normalisation statistics must be fitted only on the appropriate training subset.

For a numerical reproduction, use the original event assignments, prepared inputs, reference HDF files, spatial bases and masks, experiment settings, and selected checkpoints. Record the repository commit and installed package versions with each run.

Several defaults require particular attention:

- The manuscript uses an inundation depth threshold of 0.1 m. The current training and plotting defaults are 0.2 m; the examples above explicitly pass 0.1 m.
- The CLI defaults to three TCN channel entries (`64 64 64`). The manuscript describes four dilation levels. Use the configuration associated with the reported run or checkpoint; do not assume the CLI defaults identify that experiment.
- The depth-weighted loss coefficients must be provided from the experiment settings when that term is used.
- Inference automatically selects CUDA when available. CPU benchmarking therefore requires an explicit device-selection procedure.
- The examples document the command-line interface. They do not establish that every asset needed to reproduce the paper is included in the linked dataset.

## Scope of application

The surrogate represents the domain, terrain, and hydraulic settings reflected in its training simulations. Application to a different site or substantially different hydraulic conditions requires appropriate reference simulations, model development, and validation.

In a forecasting workflow, the supplied hydrograph would come from an upstream discharge forecast. The surrogate reconstructs hydraulic fields conditional on that hydrograph; it does not generate the upstream forecast or issue warnings. The reported hydraulic reconstruction metrics do not constitute validation of an operational warning system.

## Citation

If you use this code, please cite the accompanying manuscript:

> Qiu, Y., Zhong, Y., Hong, Y., Zhang, L., Hao, X., Huang, Y., and Yang, T. A discharge-driven low-rank framework for joint prediction of flood depth and velocity to support rapid flood warning. Manuscript.

For use of the dataset, cite the metadata and DOI provided in the [Zenodo record](https://doi.org/10.5281/zenodo.22831693). The manuscript citation will be updated when publication details become available.

## Licence and contact

The source code is distributed under the MIT License; see [LICENSE](LICENSE). Dataset reuse is governed by the licence specified in the dataset record.

For scientific or technical questions, contact Tao Yang at tao.yang@hhu.edu.cn, or open an issue in this repository. Include the command, error message, software versions, device, and relevant file structure when reporting a problem.
