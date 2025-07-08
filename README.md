# Neural Operators for the Prediction of Electromagnetic Fields in a Simulated UHF MRI Environment

This repository contains all code required to generate and reproduce the results obtained throughout the Master's thesis of Marius Bohn.

It comprises a collection of Fourier Neural Operator architectures from various authors. These models have been adapted as required in order to be compatible with the synthetic 3D UHF MRI dataset of the MAGNET4Cardiac7T project. A framework for training neural networks on the given task of predicting electromagnetic field distributions in a simulated UHF MRI environment was implemented as well.

## Contents

* The **`mrifield`** package contains own modules for training and evaluating arbitrary neural networks on the given task (`train`), scripts adapted from the magnet-pinn package for preprocessing and normalizing the data (`preprocessing`), and various (adapted) neural network architectures (`models`)
* The **`results`** folder contains metrics and plots of all relevant trainings conducted throughout this work
* A static version of the [**`magnet-pinn`**](https://github.com/MAGNET4Cardiac7T/magnet-pinn) package. It was decoupled from the corresponding repository in 05/2025 in order to obtain reproducible results and to have a stable version that continues to work with own code. It was extended to include Faraday's loss, a reworked divergence loss, and additional normalizers
* The [**`neuraloperator`**](https://github.com/neuraloperator/neuraloperator/tree/ca69be5bbf47678aecf4a666075eb6c367aee58e) repository as a submodule at commit ca69be5
* Various Slurm scripts used to process and normalize datasets, train models, and generate metrics and plots on the FAU HPC cluster

## Main use cases

* **Training**: `mrifield/train/train.py` allows to select one of the available model architectures to train by (un)commenting it and setting the corresponding hyperparameters
    * **Spectral boosting** additionally requires to define a `model_to_boost` loaded from `BOOST_CKPT` and to uncomment `lit_model_to_boost`. The latter needs to be supplied to the `lit_model` to be trained via its `model_to_boost` parameter
    * **Physics-informed training** additionally requires to supply either `DivergenceLoss()` or `FaradaysLoss()` to `lit_model` via its `pi_loss` parameter
* **Generate metrics**: `mrifield/train/metrics.py` allows to compute a variety of metrics on the test set for a trained model by (un)commenting it and supplying a checkpoint via `CKPT`
    * **Spectral boosting** additionally requires to define a `model_to_boost` loaded from `BOOST_CKPT`. It needs to be supplied to the `trained_model` to be evaluated via its `model_to_boost` parameter
    * **Physics-informed models** do not require an argument for the `pi_loss` parameter. It needs to be supplied during training only
* **Generate plots**: `mrifield/train/plots.py` allows to generate a variety of plots on the test set for a trained model in the same way as computing metrics
* **Interactive inference**: `results/inference.ipynb` contains utilities used to visualize predictions and physics-informed losses on a local copy of the test set. It was used to generate certain figures and better understand neural operators

This repository is provided for internal review only and is **not** intended to be published as it contains copies of various neural network architectures that are property of their respective authors. References and licenses are provided where applicable.