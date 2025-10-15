#!/usr/bin/env julia

using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HDF5
using Printf
using ScoreEstimation

const DATA_DIR = joinpath(@__DIR__, "data")
const DATA_FILE = joinpath(DATA_DIR, "ks_new.hdf5")
const DATASET_NAME = "timeseries"

function load_timeseries(path::AbstractString, dataset_name::AbstractString)
    h5open(path, "r") do file
        haskey(file, dataset_name) || error("Dataset $(dataset_name) not found in $(path)")
        dset = file[dataset_name]
        raw = nothing
        try
            raw = read(dset)
        finally
            close(dset)
        end

        raw === nothing && error("Failed to read dataset $(dataset_name) from $(path)")
        ndims(raw) == 2 || error("Expected a 2D dataset, got size $(size(raw))")
        n_time, n_space = size(raw)

        data = n_time >= n_space ? permutedims(raw, (2, 1)) : raw
        return Matrix(data)
    end
end

function main()
    timeseries = load_timeseries(DATA_FILE, DATASET_NAME)
    n_space, n_snapshots = size(timeseries)

    temporal_decorr, _n_uncorr, spatial_decorr =
        decorrelation_analysis(timeseries; circular_invariant=true)

    println("Kuramoto-Sivashinsky dataset summary")
    println("  data file: $(DATA_FILE)")
    println("  spatial dimensions: $(n_space)")
    println("  total snapshots: $(n_snapshots)")
    @printf("  temporal decorrelation length: %.3f\n", temporal_decorr)
    @printf("  spatial decorrelation scale: %.3f\n", spatial_decorr)
end

main()
