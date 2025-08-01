"""
ClustGen

A comprehensive toolkit for clustering-based generative modeling of dynamical systems.
This package provides tools for analyzing, clustering, and generating trajectories 
from dynamical systems using various machine learning and statistical techniques.
"""
module ScoreEstimation

# Core dependencies
using StateSpacePartitions
using LinearAlgebra
using Random
using Distributed
using SharedArrays
using StaticArrays

# Statistics and data processing
using Statistics
using StatsBase
using KernelDensity
using Optim

# I/O and visualization
using HDF5
using BSON
using Plots
using ProgressBars
using ProgressMeter
using GLMakie

# Numerical utilities
using QuadGK
using SpecialFunctions

# Deep learning
using Flux
using CUDA

# ===== Component modules =====
include("preprocessing.jl")     # Data preprocessing utilities
include("training.jl")          # Model training functions
include("sampling.jl")          # Sampling methods
include("noising_schedules.jl") # Noise schedules for diffusion models
include("io.jl")                # I/O functions for saving/loading models and data

# ===== Exported functions =====

# Clustering and preprocessing
export f_tilde, f_tilde_ssp, f_tilde_labels, generate_inputs_targets

# Model training
export train, check_loss

# Sampling methods
export sample_reverse, sample_langevin, sample_langevin_Σ

# Noise schedules
export σ_variance_exploding, g_variance_exploding

# ===== I/O functions =====
export save_variables_to_hdf5, read_variables_from_hdf5, save_current_workspace, 
       load_workspace_from_file

end # module ClustGen