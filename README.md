# ScoreEstimation: A K-means Clustering Approach to Gaussian Mixture Modeling for Score Function Estimation (KGMM)

This repository contains the Julia implementation of the KGMM algorithm, a hybrid method for accurately estimating the score function (the gradient of the logarithm of a system's steady-state probability density function).

## Overview

The ability to accurately estimate the score function is crucial for understanding the statistical properties of complex dynamical systems and for building effective stochastic reduced-order models. This is particularly relevant in fields like climate science, statistical physics, and machine learning.

Traditional methods like Gaussian Mixture Models (GMMs) can struggle when the covariance amplitude is small, leading to noise amplification in the score estimate. This project implements the **KGMM** method to address this challenge. KGMM combines GMM-based density estimation with a bisecting K-means clustering step and neural network interpolation to produce robust and accurate score function estimates, even in low-covariance regimes.

The algorithm is detailed in the paper: [KGMM: A K-means Clustering Approach to Gaussian Mixture Modeling for Score Function Estimation](https://arxiv.org/abs/2503.18054).

## The KGMM Algorithm

The core idea of KGMM is to leverage the strengths of statistical density estimation and machine learning interpolation. The process can be summarized as follows:

1.  **Clustering**: The state space is partitioned into a set of clusters using a bisecting K-means algorithm, and their centroids are computed.
2.  **Stochastic Perturbation**: Each data point from the system's trajectory is perturbed by adding Gaussian noise with a small amplitude (`σ`).
3.  **Cluster-wise Score Estimation**: The score function is estimated at each cluster centroid by averaging the normalized noise vectors of the perturbed points that fall within that cluster. This step effectively regularizes the estimate and prevents noise amplification.
4.  **Neural Network Interpolation**: A neural network is trained to learn the mapping from the cluster centroids to their corresponding score estimates, yielding a continuous and smooth approximation of the score function across the entire state space.

This hybrid approach allows for the construction of effective Langevin models (`dx/dt = F(x) + noise`) where the drift term `F(x)` is determined by the estimated score, successfully reproducing the invariant measures of the target dynamics.

## Repository Structure

The repository is organized as follows:

```
.
├── .git/
├── examples/         # Example scripts and notebooks demonstrating usage
│   └── ...
├── src/              # Source code for the KGMM algorithm
│   └── ScoreEstimation.jl
└── README.md
```

## Usage

To use the code in this repository, you will need to have Julia installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ScoreEstimation
    ```

2.  **Instantiate the Julia environment:**
    Open the Julia REPL, enter the package manager by pressing `]`, and run:
    ```julia
    pkg> activate .
    pkg> instantiate
    ```

3.  **Run an example:**
    The `examples/` directory contains scripts to reproduce the results from the paper. You can run them from the Julia REPL:
    ```julia
    include("examples/run_lorenz63.jl")
    ```

## Citation

If you use this code or the KGMM method in your research, please cite the following paper:

```bibtex
@article{giorgini2025kgmm,
  title={Kgmm: A k-means clustering approach to gaussian mixture modeling for score function estimation},
  author={Giorgini, Ludovico T and Bischoff, Tobias and Souza, Andre N},
  journal={arXiv preprint arXiv:2503.18054},
  year={2025}
}
```
