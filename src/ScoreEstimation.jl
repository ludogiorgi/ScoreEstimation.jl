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

# Public API
export train, relative_entropy

end # module
