#!/usr/bin/env julia

using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CairoMakie
using HDF5

CairoMakie.activate!()

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(@__DIR__, "data")
const FIG_PATH = joinpath(PROJECT_ROOT, "fig", "ks_alignment_heatmaps.png")
const INPUT_FILE = joinpath(DATA_DIR, "new_ks.hdf5")
const OUTPUT_FILE = joinpath(DATA_DIR, "new_ks_aligned.hdf5")
const DATASET_NAME = "timeseries"
const TARGET_INDEX = 16
const SNAPSHOT_PLOT_COUNT = 500

function load_timeseries(path::AbstractString, dataset_name::AbstractString)
    @info "Loading KS timeseries" path
    data = nothing
    h5open(path, "r") do file
        data = read(file[dataset_name])
    end
    return data
end

function align_snapshots(data::AbstractMatrix{T}; target_index::Int=TARGET_INDEX) where {T}
    n_points, n_snapshots = size(data)
    @assert 1 <= target_index <= n_points "target index $target_index outside 1:$n_points"
    aligned = similar(data)
    for j in 1:n_snapshots
        snapshot = @view data[:, j]
        _, max_idx = findmax(snapshot)
        shift = target_index - max_idx
        aligned[:, j] = circshift(snapshot, shift)
    end
    return aligned
end

function save_timeseries(path::AbstractString, dataset_name::AbstractString, data)
    @info "Writing aligned dataset" path
    mkpath(dirname(path))
    h5open(path, "w") do file
        write(file, dataset_name, data)
    end
    return nothing
end

function make_alignment_figure(original::AbstractMatrix, aligned::AbstractMatrix, fig_path::AbstractString; snapshot_count::Int=SNAPSHOT_PLOT_COUNT)
    mkpath(dirname(fig_path))
    n_snapshots = min(size(original, 2), snapshot_count)
    n_points = size(original, 1)
    original_slice = view(original, :, 1:n_snapshots)
    aligned_slice = view(aligned, :, 1:n_snapshots)
    color_min = min(minimum(original_slice), minimum(aligned_slice))
    color_max = max(maximum(original_slice), maximum(aligned_slice))

    fig = Figure(size=(1000, 700))
    snapshot_indices = 1:n_snapshots
    space_indices = 1:n_points
    ax1 = Axis(fig[1, 1], title="Original KS snapshots (first $n_snapshots)", ylabel="Space index")
    hm1 = heatmap!(ax1, snapshot_indices, space_indices, original_slice'; colorrange=(color_min, color_max), colormap=:balance)
    ax2 = Axis(fig[2, 1], title="Shift-aligned KS snapshots (first $n_snapshots)", xlabel="Snapshot index", ylabel="Space index")
    heatmap!(ax2, snapshot_indices, space_indices, aligned_slice'; colorrange=(color_min, color_max), colormap=:balance)
    Colorbar(fig[:, 2], hm1, label="Field value")
    hidexdecorations!(ax1, grid=false)
    save(fig_path, fig)
    @info "Saved comparison figure" fig_path
    return nothing
end

function main()
    original = load_timeseries(INPUT_FILE, DATASET_NAME)
    @info "Loaded dataset" size=size(original)
    aligned = align_snapshots(original)
    save_timeseries(OUTPUT_FILE, DATASET_NAME, aligned)
    make_alignment_figure(original, aligned, FIG_PATH)
    return nothing
end

main()
