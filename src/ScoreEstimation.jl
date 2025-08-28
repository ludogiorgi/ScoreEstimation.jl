# src/ScoreEstimation.jl
__precompile__()
module ScoreEstimation

using StateSpacePartitions       # Tree, StateSpacePartition
using LinearAlgebra
using Random
using Base.Threads
using Statistics

include("preprocessing.jl")

# Public API
export KGMM

end # module
