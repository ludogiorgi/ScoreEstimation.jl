# src/ScoreEstimation.jl
__precompile__()
module ScoreEstimation

using StateSpacePartitions       # Tree, StateSpacePartition
using LinearAlgebra
using Random
using Base.Threads
using Statistics
using Flux
using ProgressMeter
using KernelDensity


include("preprocessing.jl")
include("training.jl")
include("relative_entropy.jl")
include("utils.jl")

# Public API
export train, relative_entropy
export decorrelation_analysis, augment_circular_data, stack_circshifts, average_decorrelation_length
export PDFEstimate, BivariatePDFEstimate, collect_for_kde, decorrelation_metrics
export estimate_pdf_histogram, estimate_bivariate_pdf_histogram, determine_value_range, compute_averaged_pdfs

end # module
