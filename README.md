# ScoreEstimation — KGMM + NN Score Estimation in Julia

This repository provides a fast Julia implementation of KGMM and a lightweight Flux-based training wrapper to estimate the score s(x) = ∇x log p(x), its divergence ∇·s, and (optionally) its Jacobian ∂s/∂x from data. It combines a kernelized GMM (via state‑space partitioning) with a compact MLP interpolator for strong accuracy and efficiency.

## Overview

Accurate score estimation is key for statistical analysis of dynamical systems and for building stochastic reduced‑order models. This is relevant in climate science, statistical physics, and machine learning.

Traditional GMMs can struggle when the noise scale is small (variance amplification). KGMM overcomes this by estimating cluster‑wise conditional moments under Gaussian smoothing and then interpolating them with a neural network for a smooth field.

The algorithm is detailed in the paper: [KGMM: A K-means Clustering Approach to Gaussian Mixture Modeling for Score Function Estimation](https://arxiv.org/abs/2503.18054).

## Method Overview

The core idea of KGMM is to leverage the strengths of statistical density estimation and machine learning interpolation. The process can be summarized as follows:

1.  Partition the state space into clusters (bisecting K‑means on a kernelized sample).
2.  Draw noisy samples x = μ + σ z, z ∼ N(0, I), at fixed σ (no diffusion time).
3.  Accumulate cluster statistics to estimate, at each centroid, the conditional mean Ez = E[z | x] and E‖z‖².
4.  Convert moments to score/divergence via identities (see Conventions), then train a compact MLP to interpolate the score over the state space.

This hybrid approach yields robust score fields and enables construction of effective Langevin models (`dx = F(x)dt + s dW`) where `F` is informed by the estimated score.

## Repository Structure

The repository is organized as follows:

```
.
├── src/
│   ├── ScoreEstimation.jl   # module; exports KGMM, train
│   ├── preprocessing.jl     # KGMM implementation
│   └── training.jl          # Flux training (DSM + KGMM preprocessing)
├── scripts/                 # Example scripts (reduced.jl, triad.jl)
├── test/                    # Pkg.test entry
│   └── runtests.jl
├── figures/                 # Generated figures from scripts
├── experiments/             # Playground notebooks/scripts (optional)
└── README.md
```

## Installation

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

Optionally, to use a GPU with Flux set `use_gpu=true` in the training calls and ensure CUDA.jl is functional on your system.

## Quickstart

Given observations `obs::AbstractMatrix` of size `(D, N)` and a scalar smoothing scale `σ`:

```julia
using ScoreEstimation

D, N = size(obs)
σ = 0.1

# Train with KGMM preprocessing (recommended)
nn, losses, _, div_fn, jac_fn, kgmm = ScoreEstimation.train(
    obs; preprocessing=true, σ=σ, neurons=[128, 128],
    n_epochs=200, batch_size=1024, lr=1e-3, use_gpu=false, verbose=true,
    kgmm_kwargs=(prob=0.001, conv_param=1e-2, i_max=100, show_progress=false),
    divergence=true, probes=1, jacobian=true)

# Score, divergence, and Jacobian closures
sθ = X -> -nn(Float32.(X)) ./ Float32(σ)
∇·sθ = X -> div_fn(X)  # returns (1, B)
J_s = X -> jac_fn(X)   # returns (D, D, B)
```

Alternatively, raw DSM training (no preprocessing):

```julia
nn, losses, _, div_fn, jac_fn, _ = ScoreEstimation.train(
    obs; preprocessing=false, σ=σ, neurons=[128, 128],
    n_epochs=200, batch_size=1024, lr=1e-3, use_gpu=false, verbose=true,
    divergence=true, probes=1, jacobian=true)
```

### Conventions

This repository uses a single, standard smoothing convention

- x = μ + σ z, z ∼ N(0, I)
- s(x) = ∇x log p(x) = − E[z | x] / σ
- ∇·s(x) = (E[‖z‖² | x] − d)/σ² − ‖s(x)‖²

The NN models learn ε̂(x) ≈ E[z | x] and are converted to score/divergence via the identities above.
When `jacobian=true`, the training wrapper also exposes a closure `jac_fn(X)` that returns the exact Jacobian of the score at batch columns `X` with shape `(D, D, B)`. It is computed efficiently via reverse-mode accumulation of Jacobian rows in `D` passes (works on CPU and GPU).

### Progress Bars

When `verbose=true`, training loops display a progress bar (ProgressMeter.jl) for epochs.

## Examples (in `scripts/`)

- 1D reduced example (with analytic comparisons):
  - Run: `julia --project=. scripts/reduced.jl`
  - Produces a two‑panel figure comparing analytic, KGMM and NN results, plus a second figure comparing NN+KGMM vs NN (DSM).

- 3D triad system (no analytic reference):
  - Run: `julia --project=. scripts/triad.jl`
  - Trains two models on the normalized data: one with KGMM preprocessing and one with raw DSM.
  - Generates two figures in `figures/`:
    - `triad_pdfs.png`: three panels (X/Y/Z) showing the true marginal PDFs (from data) vs PDFs generated by the two learned score models via Langevin simulation (NN+KGMM and NN (DSM)).
    - `triad_divergences.png`: three panels (X/Y/Z) comparing the estimated divergences `∇·sθ` evaluated at samples from the dataset for the two models.

Both scripts use the unified smoothing convention and the repository’s training wrapper. GPU usage is optional (`use_gpu=true` in the training calls if CUDA is available).

## KGMM API

To compute cluster centers and moments directly (without training):

```julia
res = ScoreEstimation.KGMM(σ, obs; prob=0.001, conv_param=1e-2, i_max=100, show_progress=false)
centers     = res.centers          # (D, C)
score       = res.score            # (D, C)
divergence  = res.divergence       # (C,)
counts      = res.counts           # (C,)
```

### Tips
- Use the returned `counts` as sample weights when supervising on centers (the training wrapper does this automatically).
- Increase `i_max` or tighten `conv_param` for stricter EMA convergence.

## Testing

This package includes a basic test suite.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Performance Notes

- Use Float32 inputs and keep data on-device for throughput.
- Prefer preprocessing=true for small/noisy datasets; it stabilizes NN interpolation.
- For GPU use, ensure CUDA.functional() returns true and pass `use_gpu=true`.

## Citation

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
