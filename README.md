# Physics-Informed Fourier Neural Operators for Electromagnetic Field Prediction in Ultra-High-Field MRI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion code for the paper:

> **Physics-Informed Fourier Neural Operators for Electromagnetic Field Prediction in Ultra-High-Field MRI**
> Andrzej Dulny, Marius Bohn, Farzad Jabbarigargari, Andreas Hotho, Laura Maria Schreiber, Maxim Terekhov, Anna Krause
> *ML4EMS Workshop, 2026 (to appear)*

Ultra-high-field (7 T) MRI requires patient-specific assessment of the specific absorption rate (SAR) to ensure safety, but conventional FDTD simulations are too slow for real-time clinical workflows. This repository evaluates **Fourier Neural Operator (FNO) variants** as fast surrogate models for predicting 3D electromagnetic field distributions in a simulated UHF MRI environment with eight transmit dipoles, and compares them against a 3D U-Net baseline. It further implements **physics-informed loss terms** derived from Maxwell's equations — the magnetic divergence constraint (∇·**B** = 0) and Faraday's law (∇×**E** + jω**B** = 0) — as well as **spectral boosting**, which together yield up to 15.5 % accuracy improvements at zero additional inference cost.

The code was developed by [Marius Bohn](https://github.com/mariusbohn) as part of his Master's thesis at CAIDAS, University of Würzburg, within the [MAGNET4Cardiac7T](https://www.uni-wuerzburg.de/en/magnet4cardiac7t/) project.

## Repository structure

| Path | Description |
|---|---|
| `mrifield/` | Main package: training/evaluation framework (`train`), data preprocessing and normalization (`preprocessing`), and adapted neural network architectures (`models`) |
| `mrifield/train/train.py` | Model training entry point |
| `mrifield/train/metrics.py`, `metrics_sar.py` | Test-set metrics for field prediction and SAR estimation |
| `mrifield/train/plots.py`, `plots_sar.py` | Plot generation for trained models |
| `magnet-pinn/` | Static copy of the [magnet-pinn](https://github.com/MAGNET4Cardiac7T/magnet-pinn) package (decoupled 05/2025 for reproducibility), extended with Faraday's loss, a reworked divergence loss, and additional normalizers |
| `neuraloperator/` | The [neuraloperator](https://github.com/neuraloperator/neuraloperator) library as a git submodule, pinned at commit `ca69be5` (provides FNO, TFNO, UNO) |
| `results/` | Metrics and plots for all reported trainings, plus `inference.ipynb` for interactive visualization of predictions and physics-informed losses |
| `tr_/mx_/pl_/pr_/nm_mrifield` | Slurm job scripts for training, metrics, plots, preprocessing, and normalization on an HPC cluster |

## Installation

```bash
git clone --recurse-submodules https://github.com/badulion/fno_magnet.git
cd fno_magnet
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `neuraloperator` and `magnet-pinn` directories are used directly via `PYTHONPATH` (see the Slurm scripts):

```bash
export PYTHONPATH=$(pwd)/magnet-pinn:$(pwd)/neuraloperator:$PYTHONPATH
```

## Usage

### Training

`mrifield/train/train.py` selects the model architecture to train by (un)commenting it and setting the corresponding hyperparameters:

```bash
python3.11 -m mrifield.train.train
```

- **Physics-informed training**: supply `DivergenceLoss()` or `FaradaysLoss()` to `lit_model` via its `pi_loss` parameter.
- **Spectral boosting**: define a `model_to_boost` loaded from `BOOST_CKPT`, uncomment `lit_model_to_boost`, and pass it to the `lit_model` via its `model_to_boost` parameter.

### Evaluation

```bash
python3.11 -m mrifield.train.metrics   # test-set metrics (set CKPT to the checkpoint)
python3.11 -m mrifield.train.plots     # test-set plots
```

- Spectral-boosted models additionally need `model_to_boost` (from `BOOST_CKPT`) passed to the evaluated model.
- Physics-informed models need `pi_loss` **only during training** — no argument is required for evaluation.

### SLURM

The extensionless scripts in the repository root submit the corresponding jobs on an HPC cluster (adjust the `#SBATCH` directives to your site):

```bash
sbatch tr_mrifield   # training
sbatch mx_mrifield   # metrics
sbatch pl_mrifield   # plots
sbatch pr_mrifield   # test-set preprocessing
sbatch nm_mrifield   # normalization
```

## Third-party code

This repository bundles adapted model implementations that retain their original licenses:

| Component | Source | License |
|---|---|---|
| `mrifield/models/unet3d` | [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) (A. Wolny) | MIT |
| `mrifield/models/ffno` | [factorized-fno](https://github.com/alasdairtran/fourierflow) (A. Tran) | MIT |
| `mrifield/models/afno` | [FourCastNet](https://github.com/NVlabs/FourCastNet) (NVIDIA) | BSD-3-Clause |
| `magnet-pinn/` | [MAGNET4Cardiac7T/magnet-pinn](https://github.com/MAGNET4Cardiac7T/magnet-pinn) | GPL-3.0 |
| `neuraloperator/` (submodule) | [neuraloperator](https://github.com/neuraloperator/neuraloperator) | MIT |

U-FNO ([Wen et al., 2022](https://github.com/gegewen/ufno)) and the Convolutional Neural Operator ([Raonić et al., 2023](https://github.com/camlab-ethz/ConvolutionalNeuralOperator)) were explored during the thesis but are not part of the paper; their code is not redistributed here — please refer to the original repositories.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{dulny2026fno_magnet,
  title     = {Physics-Informed Fourier Neural Operators for Electromagnetic Field Prediction in Ultra-High-Field {MRI}},
  author    = {Dulny, Andrzej and Bohn, Marius and Jabbarigargari, Farzad and Hotho, Andreas and Schreiber, Laura Maria and Terekhov, Maxim and Krause, Anna},
  booktitle = {ML4EMS Workshop},
  year      = {2026},
  note      = {to appear}
}
```

## Acknowledgments

The project underlying this publication was funded by the German Federal Ministry of Education and Research under the grant number 16DKWN0099B (MAGNET4Cardiac7T). The responsibility for the content of this publication lies with the authors.
