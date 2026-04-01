# src/ScoreEstimation.jl
__precompile__()
module ScoreEstimation

using StateSpacePartitions       # Tree, StateSpacePartition
using LinearAlgebra
using SparseArrays
using Random
using Base.Threads
using Statistics
using BSON
using Flux
using HDF5
using ProgressMeter
using KernelDensity

const _CUDA_MODULE = Ref{Any}(nothing)
const _CUDA_LOAD_ATTEMPTED = Ref(false)

function _cuda_module()
    if _CUDA_LOAD_ATTEMPTED[] && _CUDA_MODULE[] === nothing
        return nothing
    end
    if _CUDA_MODULE[] !== nothing
        return _CUDA_MODULE[]
    end

    _CUDA_LOAD_ATTEMPTED[] = true
    try
        @eval import CUDA
        _CUDA_MODULE[] = CUDA
    catch err
        @debug "CUDA could not be loaded; GPU features will be disabled" exception=(err, catch_backtrace())
        _CUDA_MODULE[] = nothing
    end
    return _CUDA_MODULE[]
end

function _cuda_functional()
    cuda = _cuda_module()
    return cuda !== nothing && Base.invokelatest(cuda.functional)
end

function _to_device(x; use_gpu::Bool=false)
    if use_gpu && _cuda_functional()
        cuda = _cuda_module()
        return Base.invokelatest(cuda.cu, x)
    end
    return x
end

_device_name(use_gpu::Bool) = use_gpu && _cuda_functional() ? "GPU" : "CPU"


include("preprocessing.jl")
include("io.jl")
include("phi_sigma.jl")
include("ensemble_langevin.jl")
include("training.jl")
include("relative_entropy.jl")
include("utils.jl")
include("brusselator.jl")

# Public API
export kgmm
export train, relative_entropy
export save_model, load_model, save_variables_to_hdf5, read_variables_from_hdf5, load_to_workspace
export ScoreModel, LinearScoreSDE, evolve_score_langevin, evolve_score_langevin_snapshots
export evolve_affine_score_langevin, evolve_affine_score_langevin_snapshots
export evolve_interpolated_langevin_snapshots
export BrusselatorConfig, fixed_point, brusselator_drift!, brusselator_sigma
export integrate_brusselator_timeseries, integrate_brusselator_stationary
export estimate_uncorrelated_stride, subsample_uncorrelated, build_brusselator_diagnostics
export compare_brusselator_diagnostics
export sparse_generator, compute_cluster_centers_and_pi, compute_generator_cdot
export compute_score_position_matrix, estimate_phi_sigma, generator_correlation
export decorrelation_analysis, augment_circular_data, stack_circshifts, average_decorrelation_length
export PDFEstimate, BivariatePDFEstimate, collect_for_kde, decorrelation_metrics
export estimate_pdf_histogram, estimate_bivariate_pdf_histogram, determine_value_range, compute_averaged_pdfs

end # module
