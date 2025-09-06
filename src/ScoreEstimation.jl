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


include("preprocessing.jl")
include("training.jl")

# Public API
export KGMM
export train

end # module
