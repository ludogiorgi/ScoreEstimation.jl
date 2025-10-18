using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# Stand-alone driver that mirrors the reduced 1D workflow in `scripts/reduced/`
# but targets the 32-mode projection of the Kuramoto–Sivashinsky dataset. The
# script performs: data loading, stride-based subsampling, score training via
# `ScoreEstimation.train`, Langevin sampling with FastSDE, and
# PDF/relative-entropy diagnostics in physical space.

# ============================================================================
# 🚀 MULTI-THREADING SETUP - READ THIS IF YOU SEE "n_threads = 1"
# ============================================================================
#
# PROBLEM: Pressing Ctrl+Shift+Enter shows "n_threads = 1"?
#
# SOLUTION (takes 10 seconds):
#   1. Press: Ctrl+Shift+P (or Cmd+Shift+P on Mac)
#   2. Type: "Julia: Kill Julia" and press Enter
#   3. Run this file again with Ctrl+Shift+Enter
#   4. You should now see "threads = 16" (or your CPU count)
#
# WHY? Julia's thread count is set when it starts. You need to restart
# the Julia REPL for the new threading settings to take effect.
#
# ALTERNATIVE: Use the build task instead:
#   - Press Ctrl+Shift+B (or Cmd+Shift+B)
#   - This always uses all available threads
#
# OR: Use the shell script (always works):
#   ./run_experiment.sh auto --langevin-ens=64
#
# More details: See SETUP.md and VSCODE_THREADING_FIX.md
# ============================================================================

using CairoMakie
using BSON
using TOML

using FFTW
using FastSDE
using Flux
using HDF5
using LinearAlgebra
using Random
using Printf
using ScoreEstimation
using Statistics

const KS_ROOT = @__DIR__
const KS_DATA_DIR = joinpath(KS_ROOT, "data")
const KS_RUNS_DIR = joinpath(KS_ROOT, "runs")
const KS_DATAFILE = joinpath(KS_DATA_DIR, "ks_new.hdf5")
const KS_OUTPUT_H5 = joinpath(KS_DATA_DIR, "ks_results.h5")

CairoMakie.activate!()

# Configuration container that keeps all runtime parameters together. These
# defaults (stride, σ, network shape, etc.) reflect the heuristics used in the
# reduced 1D scripts but scaled up to 32-dimensional latent coordinates.
struct KSConfig
    stride::Int
    max_snapshots::Int
    n_modes::Int
    seed::Int
    sigma::Float64
    neurons::Vector{Int}
    n_epochs::Int
    batch_size::Int
    lr::Float64
    kgmm_kwargs::NamedTuple
    train_max::Int
    langevin_boundary::Float64
    langevin_dt::Float64
    langevin_steps::Int
    langevin_resolution::Int
    langevin_ens::Int
    langevin_burnin::Int
    plot_window::Int
    kde_max_samples::Int
    rel_entropy_points::Int
    normalize_mode::Symbol
    train_copies::Int
    dataset_key::Union{Nothing,String}
end

# Lightweight CLI parser similar to the reduced scripts. We avoid hard
# dependencies (e.g. ArgParse) to keep the entrypoint self-contained.
function parse_args()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--")
            key = lowercase(arg[3:end])
            if i == length(ARGS) || startswith(ARGS[i + 1], "--")
                args[key] = "true"
            else
                args[key] = ARGS[i + 1]
                i += 1
            end
        else
            @warn "Ignoring positional argument" arg
        end
        i += 1
    end
    return args
end

function parse_vector(arg::String)
    isempty(strip(arg)) && return Int[]
    return [parse(Int, strip(x)) for x in split(arg, ",") if !isempty(strip(x))]
end

"""
    build_config() -> KSConfig

Turn parsed CLI flags into a `KSConfig`. Mirrors the ergonomics of
`reduced_compute.jl` so automated experiments can tune noise level, network
depth, and Langevin integrator without editing the source.
"""
function build_config()
    args = parse_args()

    getint(key, default) = parse(Int, get(args, key, string(default)))
    getfloat(key, default) = parse(Float64, get(args, key, string(default)))
    getbool(key; default=false) = get(args, key, default ? "true" : "false") == "true"

    stride = getint("stride", 10)                             # Temporal subsampling interval when reading KS data
    max_snapshots = getint("max-snapshots", 0)               # Hard cap on temporal snapshots (0 = no cap)
    n_modes = getint("n-modes", 8)                           # Spatial stride for mode subsampling (1 = keep all grid points)
    seed = getint("seed", 2024)                              # RNG seed for reproducibility
    sigma = getfloat("sigma", 0.1)                           # Noise level σ used in ε-training / score scaling

    neurons = [128, 64]
    n_epochs = getint("n-epochs", 100)                       # Training epochs for ε-network
    batch_size = getint("batch-size", 64)                    # Batch size for Flux DataLoader
    lr = getfloat("lr", 1e-3)                                # Learning rate for Adam optimizer

    prob = getfloat("prob", 1e-4)                            # kgmm: probability threshold for cluster pruning / weighting
    conv_param = getfloat("conv-param", 1e-3)                # kgmm: convergence tolerance
    i_max = getint("imax", 120)                              # kgmm: max iterations
    show_progress = getbool("show-progress")                 # Verbose kgmm progress display toggle
    kgmm_kwargs = (prob=prob, conv_param=conv_param, i_max=i_max, show_progress=show_progress)

    train_max = getint("train-max", 0)                       # Optional cap on latent samples used for training (0 = all)
    langevin_boundary = getfloat("langevin-boundary", 10.0)  # Hard wall in normalized space (per coordinate)
    langevin_dt = getfloat("langevin-dt", 0.0025)            # Time step for Langevin Euler–Maruyama integration
    langevin_steps = getint("langevin-steps", 400000)        # Total Langevin steps (including burn-in)
    langevin_resolution = getint("langevin-resolution", 40)  # Recording interval after burn-in (thinning factor)
    langevin_ens = getint("langevin-ens", 100)               # Number of parallel ensemble members during sampling
    langevin_burnin = getint("langevin-burnin", 0)           # Discard initial steps (thermalization)

    plot_window = getint("plot-window", 1000)                # Max length of time series segment displayed in subplot 1
    kde_max_samples = getint("kde-max-samples", 0)           # Upper bound on samples fed to KDE (0 = use all)
    rel_entropy_points = getint("rel-entropy-points", 2048)  # Quadrature points for relative entropy estimation

    normalize_mode_str = lowercase(get(args, "normalize-mode", "per_dim"))
    normalize_mode = normalize_mode_str == "max" ? :max : :per_dim
    train_copies = max(1, getint("train-copies", 8))
    dataset_key_str = strip(get(args, "dataset-key", ""))
    dataset_key = isempty(dataset_key_str) ? nothing : dataset_key_str

    return KSConfig(stride, max_snapshots, n_modes, seed, sigma, neurons, n_epochs, batch_size, lr,
        kgmm_kwargs, train_max, langevin_boundary, langevin_dt, langevin_steps, langevin_resolution,
        langevin_ens, langevin_burnin, plot_window, kde_max_samples,
        rel_entropy_points, normalize_mode, train_copies, dataset_key)
end


"""
    load_ks_data(path; stride, max_snapshots, dataset_key=nothing) -> (field, Δt?)

Read the historical KS simulation stored in `ks_old.hdf5`. We lazily stride the
time axis to control memory and optionally cap the number of snapshots so the
downstream subsampling/training pipeline remains tractable.
"""
function load_ks_data(path::AbstractString; stride::Int, max_snapshots::Int, dataset_key::Union{Nothing,String}=nothing)
    @info "Loading KS data" path stride max_snapshots dataset_key
    data = nothing
    delta_t = nothing
    h5open(path, "r") do file
        target_key = dataset_key
        if target_key === nothing
            dataset_candidates = String[]
            for name in keys(file)
                obj = file[name]
                try
                    if obj isa HDF5.Dataset
                        push!(dataset_candidates, String(name))
                    end
                finally
                    close(obj)
                end
            end
            preferred_order = ("u", "timeseries", "field", "data")
            for candidate in preferred_order
                if candidate in dataset_candidates
                    target_key = candidate
                    break
                end
            end
            if target_key === nothing && !isempty(dataset_candidates)
                target_key = first(dataset_candidates)
            end
        else
            haskey(file, target_key) || error("Dataset key $(target_key) not found in $(path)")
        end

        target_key === nothing && error("No datasets found in $(path)")
        dset = file[target_key]
        dims = size(dset)
        time_axis = if dims[1] >= dims[end]
            1
        else
            2
        end
        spatial_axis = 3 - time_axis
        n_time = dims[time_axis]
        idx_time = 1:stride:n_time
        if max_snapshots > 0 && length(idx_time) > max_snapshots
            last_index = first(idx_time) + stride * (max_snapshots - 1)
            idx_time = first(idx_time):stride:min(last_index, n_time)
        end
        @info "Reading dataset selection" nsnapshots=length(idx_time) spatial=dims[spatial_axis]
        data = if time_axis == 1
            Array(dset[idx_time, :])'
        else
            Array(dset[:, idx_time])
        end
        if haskey(file, "Δt")
            delta_t = read(file, "Δt")
        end
        close(dset)
    end
    data = convert(Matrix{Float64}, data)
    return data, delta_t
end

"""
    select_modes(data, stride) -> (Matrix, Vector{Int})

Subsample the spatial grid by keeping every `stride`-th coordinate. The
returned matrix contains the centered values of the retained coordinates while
the companion vector lists the selected indices for reconstruction.
"""
function select_modes(data::AbstractMatrix, stride::Int)
    stride <= 0 && error("n_modes (stride) must be positive; got $(stride)")
    nx, _ = size(data)
    indices = collect(1:stride:nx)
    isempty(indices) && error("Mode subsampling produced no coordinates; check n_modes and grid size")
    return data[indices, :], indices
end

"""
    build_circshifted_modes(data, stride, copies) -> (centered, indices, mean_field)

Construct the centered, mode-restricted dataset produced by
`select_modes(stack_circshifts(data, copies) .- mean; stride)` without
materialising the full stacked matrix. Returns the centered coordinates,
the retained indices, and the per-row mean of the stacked field.
"""
function build_circshifted_modes(data::AbstractMatrix{T}, stride::Int, copies::Int) where {T<:AbstractFloat}
    copies = max(copies, 1)
    stride <= 0 && error("n_modes (stride) must be positive; got $(stride)")
    D, N = size(data)
    indices = collect(1:stride:D)
    isempty(indices) && error("Mode subsampling produced no coordinates; check n_modes and grid size")

    row_sums = Vector{Float64}(undef, D)
    @inbounds for r in 1:D
        row_sums[r] = sum(@view data[r, :])
    end

    denom = Float64(N) * Float64(copies)
    mean_field = Vector{T}(undef, D)
    @inbounds for r in 1:D
        acc = 0.0
        for shift in 0:(copies - 1)
            acc += row_sums[mod1(r - shift, D)]
        end
        mean_field[r] = T(acc / denom)
    end

    total_cols = N * copies
    centered = Matrix{T}(undef, length(indices), total_cols)
    @inbounds for (block_idx, shift) in enumerate(0:(copies - 1))
        col_start = (block_idx - 1) * N + 1
        col_end = block_idx * N
        col_range = col_start:col_end
        for (row_pos, row_idx) in enumerate(indices)
            src_row = mod1(row_idx - shift, D)
            src = view(data, src_row, :)
            dest = view(centered, row_pos, col_range)
            μ = mean_field[row_idx]
            @inbounds @simd for j in 1:N
                dest[j] = src[j] - μ
            end
        end
    end

    return centered, indices, mean_field
end

"""
    normalize_data(X, mode) -> (Xn, means, stds)

Standardize latent coordinates according to `mode`. When `mode == :per_dim`
each coordinate is divided by its own standard deviation. When `mode == :max`
all coordinates are scaled by the largest per-dimension standard deviation,
so the most variable component attains unit variance while others remain
subunit. Zero-variance coordinates fall back to `eps(Float64)` to avoid
division by zero.
"""
function normalize_data(X::AbstractMatrix{T}, mode::Symbol; inplace::Bool=false) where {T<:Real}
    means64 = vec(mean(X; dims=2))
    raw_stds64 = vec(std(X; dims=2))
    raw_stds64 = map(x -> x > 0 ? x : eps(Float64), raw_stds64)
    stds64 = if mode == :max
        max_std = maximum(raw_stds64)
        scale = max(max_std, eps(Float64))
        fill(scale, length(raw_stds64))
    else
        raw_stds64
    end

    means = T.(means64)
    stds = T.(stds64)
    Xn = inplace ? X : similar(X)

    d, n = size(X)
    @inbounds for i in 1:d
        μ = means[i]
        σ = stds[i]
        σ = σ == 0 ? T(eps(T)) : σ
        invσ = T(1) / σ
        src = view(X, i, :)
        dest = inplace ? src : view(Xn, i, :)
        @simd for j in 1:n
            dest[j] = (src[j] - μ) * invσ
        end
    end

    return Xn, means, stds
end

function unnormalize_data(Xn::AbstractMatrix, means::AbstractVector, stds::AbstractVector)
    means_mat = reshape(means, :, 1)
    stds_mat = reshape(stds, :, 1)
    return (Xn .* stds_mat) .+ means_mat
end

# average_decorrelation_length is now provided by ScoreEstimation.average_decorrelation_length
# from utils.jl - no need to redefine it here

"""
    train_score_model(Xn, cfg)

Thin wrapper around `ScoreEstimation.train`. We reuse the same kgmm-assisted
preprocessing branch that powers the reduced experiments and keep the API fully
typed so experiment scripts can introspect losses/cluster counts.
"""
function train_score_model(Xn::AbstractMatrix, cfg::KSConfig)
    obs = eltype(Xn) <: Float32 ? Xn : Float32.(Xn)
    Random.seed!(cfg.seed)
    nn, losses, _, _, _, res, _ = ScoreEstimation.train(
        obs;
        preprocessing=true,
        σ=cfg.sigma,
        neurons=cfg.neurons,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        use_gpu=false,
        verbose=true,
        kgmm_kwargs=cfg.kgmm_kwargs,
        divergence=false,
    )
    return nn, losses, res
end

"""
    create_batched_drift_nn(nn, σ)

Wrap the ε-network into a FastSDE-compatible drift. The conversion to Float32
matches the training precision; we reuse a per-thread cached buffer to avoid
allocating in the tight Langevin loop.
"""
function create_batched_drift_nn(nn, sigma::Real)
    inv_sigma = 1.0 / Float64(sigma)
    buf_store = [Matrix{Float32}(undef, 0, 0) for _ in 1:Threads.nthreads()]

    @inline function get_buffer(dim::Int, ncols::Int)
        tid = Threads.threadid()
        buf = buf_store[tid]
        if size(buf, 1) != dim || size(buf, 2) != ncols
            buf = Matrix{Float32}(undef, dim, ncols)
            buf_store[tid] = buf
        end
        return buf
    end

    function drift_batched!(DU, U, p, t)
        if U isa AbstractVector
            dim = length(U)
            ncols = 1
        else
            dim, ncols = size(U)
        end

        buf = get_buffer(dim, ncols)

        if U isa AbstractVector
            @inbounds for i in 1:dim
                buf[i, 1] = Float32(U[i])
            end
        else
            @inbounds for j in 1:ncols
                for i in 1:dim
                    buf[i, j] = Float32(U[i, j])
                end
            end
        end

        Y = nn(buf)

        if DU isa AbstractVector
            @inbounds for i in 1:dim
                DU[i] = -Float64(Y[i, 1]) * inv_sigma
            end
        else
            @inbounds for j in 1:ncols
                for i in 1:dim
                    DU[i, j] = -Float64(Y[i, j]) * inv_sigma
                end
            end
        end
        return nothing
    end

    function drift_batched!(DU, U, t)
        drift_batched!(DU, U, nothing, t)
    end

    return drift_batched!
end

"""
    sample_langevin(dim, nn, cfg) -> Matrix{Float64}

Primary Langevin driver using the FastSDE integrator. We record at the desired
resolution, drop a burn-in window (including the initial condition), and return
flattened samples in the normalized latent space.

Note: FastSDE.evolve with n_ens > 1 uses Threads.@threads internally to parallelize
ensemble members. Each thread calls the drift function with a vector (single trajectory).
"""
function sample_langevin(dim::Int, nn, cfg::KSConfig)
    # Create drift function that handles vectors (single trajectories)
    # FastSDE's ensemble mode calls this once per ensemble member in parallel threads
    drift_batched! = create_batched_drift_nn(nn, cfg.sigma)

    init = zeros(Float64, dim)
    dt = cfg.langevin_dt
    resolution = max(cfg.langevin_resolution, 1)
    boundary_radius = cfg.langevin_boundary
    if boundary_radius <= 0
        error("langevin_boundary must be positive; got $(boundary_radius)")
    end
    lower = -boundary_radius
    upper = boundary_radius

    @info "Starting Langevin sampling" n_ens=cfg.langevin_ens steps=cfg.langevin_steps resolution=resolution n_threads=Threads.nthreads()

    # Use evolve with n_ens for parallel ensemble integration
    # Note: drift_batched! is thread-safe because nn inference is reentrant in Flux
    samples_full = FastSDE.evolve(
        init,
        dt,
        cfg.langevin_steps,
        drift_batched!,
        sqrt(2.0);
        timestepper=:euler,
        resolution=resolution,
        n_ens=cfg.langevin_ens,
        manage_blas_threads=true,
        boundary=(lower, upper),
        flatten=true,
    )

    samples_mat = Array(samples_full)
    dim_out, total_cols = size(samples_mat)
    @info "Langevin sampling completed" output_shape=(dim_out, total_cols)

    if total_cols == 0
        return zeros(Float64, dim_out, 0)
    end

    # When flatten=true, trajectories are concatenated: [traj1, traj2, ..., trajN]
    # Each trajectory has (langevin_steps / resolution) + 1 snapshots
    snapshots_per_traj = div(cfg.langevin_steps, resolution) + 1

    # Calculate burn-in: drop initial snapshots from each trajectory
    burnin_snapshots = ceil(Int, cfg.langevin_burnin / resolution)
    drop_snapshots_per_traj = min(snapshots_per_traj, burnin_snapshots + 1)  # +1 for initial condition

    # Build indices to keep: skip burn-in from each trajectory
    keep_indices = Int[]
    for ens_idx in 0:(cfg.langevin_ens - 1)
        traj_start = ens_idx * snapshots_per_traj + 1
        keep_start = traj_start + drop_snapshots_per_traj
        keep_end = (ens_idx + 1) * snapshots_per_traj
        if keep_start <= keep_end
            append!(keep_indices, keep_start:keep_end)
        end
    end

    if isempty(keep_indices)
        @warn "All samples removed during burn-in" burnin_snapshots drop_snapshots_per_traj snapshots_per_traj
        return zeros(Float64, dim_out, 0)
    end

    @info "Burn-in removed" kept_samples=length(keep_indices) dropped_per_trajectory=drop_snapshots_per_traj

    return samples_mat[:, keep_indices]
end

"""
    assemble_physical_subset(X, mean_field, indices) -> Matrix

Convert centered snapshots restricted to `indices` back into physical-space
values by adding the spatial mean of the retained coordinates. The result is a
matrix containing only the sampled locations, matching the latent dimensionality.
"""
function assemble_physical_subset(X::AbstractMatrix, mean_field::AbstractVector, indices::AbstractVector{Int})
    isempty(indices) && error("Cannot assemble physical subset with no retained indices")
    subset_mean = reshape(mean_field[indices], :, 1)
    return subset_mean .+ X
end

# collect_for_kde, decorrelation_metrics now provided by ScoreEstimation.jl (utils.jl)

"""
    spatial_decorrelation_length(field; max_samples=4096) -> Float64

Wrapper for computing spatial decorrelation scale using circular invariance.
This uses the optimized implementation from utils.jl.
Note: max_samples parameter is kept for API compatibility but subsampling
is handled internally by the utils.jl implementation.
"""
function spatial_decorrelation_length(field::AbstractMatrix; max_samples::Int=4096)
    # Use the circular invariant version from utils.jl
    # (max_samples is handled internally in _compute_spatial_decorrelation_scale)
    _, _, decorr_scale = ScoreEstimation.decorrelation_analysis(field; circular_invariant=true)
    return decorr_scale
end

# PDFEstimate, BivariatePDFEstimate now provided by ScoreEstimation.jl (utils.jl)

# estimate_pdf_histogram, estimate_bivariate_pdf_histogram, determine_value_range, compute_averaged_pdfs
# now provided by ScoreEstimation.jl (utils.jl)

"""
    build_plot(series_emp, series_gen, X, X_gen, kde_emp, kde_gen,
               avg_uni_emp, avg_uni_gen, avg_biv_emp, avg_biv_gen,
               rel_ent, cfg, cluster_count, delta_t, observable_idx)

Create a publication-style diagnostic figure with 8 panels (4 rows × 2 columns):
- Row 1: Time series (left) and univariate PDF comparison (right)
- Rows 2-4: Bivariate PDFs for distances 1, 2, 3 (observed left, generated right)
"""
function build_plot(series_emp::AbstractVector, series_gen::AbstractVector,
                    X::AbstractMatrix, X_gen::AbstractMatrix,
                    kde_emp::PDFEstimate,
                    kde_gen::PDFEstimate,
                    avg_uni_emp::PDFEstimate,
                    avg_uni_gen::PDFEstimate,
                    avg_biv_emp::Vector{BivariatePDFEstimate},
                    avg_biv_gen::Vector{BivariatePDFEstimate},
                    rel_ent::Real, cfg::KSConfig, cluster_count::Integer,
                    delta_t, observable_idx::Int;
                    save_path::Union{Nothing,String}=nothing)
    save_path === nothing && error("build_plot requires a save_path when global figure directory is disabled")
    mkpath(dirname(save_path))
    fig_path = save_path
    mode_count = size(X, 1)
    clusters_tag = max(cluster_count, 0)
    stride_tag = cfg.n_modes

    # Create figure with 4 rows × 2 columns layout
    fig = Figure(size=(1800, 3600), fontsize=28)

    # Get axis limits from observed data
    uni_xlim = (minimum(avg_uni_emp.x), maximum(avg_uni_emp.x))
    uni_ylim = (0.0, maximum(avg_uni_emp.density) * 1.1)

    # Get bivariate limits from observed data
    biv_xlim = (minimum(avg_biv_emp[1].x), maximum(avg_biv_emp[1].x))
    biv_ylim = (minimum(avg_biv_emp[1].y), maximum(avg_biv_emp[1].y))

    # Find max density for consistent colorbar across all bivariate plots
    biv_density_max = 0.0
    for biv_est in vcat(avg_biv_emp, avg_biv_gen)
        biv_density_max = max(biv_density_max, maximum(biv_est.density))
    end
    biv_density_max = max(biv_density_max, eps(Float64))

    # Row 1, Column 1: Time series
    time_ax = Axis(fig[1, 1];
        xlabel="t",
        ylabel="u(x*, t)",
        title=@sprintf("Time series (idx=%d)", observable_idx),
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28)
    tshow = minimum((cfg.plot_window, length(series_emp), length(series_gen)))
    dt_plot = isnothing(delta_t) ? 1.0 : delta_t * cfg.stride
    time_vector = (0:(tshow - 1)) .* dt_plot
    lines!(time_ax, time_vector, series_emp[1:tshow]; color=:steelblue, linewidth=2.0, label="Observed")
    lines!(time_ax, time_vector, series_gen[1:tshow]; color=:tomato, linewidth=2.0, linestyle=:dash, label="Generated")
    axislegend(time_ax; position=:rt, framevisible=true, backgroundcolor=:white, labelsize=24)

    # Row 1, Column 2: Univariate PDF comparison
    univariate_ax = Axis(fig[1, 2];
        xlabel="u",
        ylabel="PDF",
        title="Averaged Univariate PDF",
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28)
    xlims!(univariate_ax, uni_xlim)
    ylims!(univariate_ax, uni_ylim)
    lines!(univariate_ax, avg_uni_emp.x, avg_uni_emp.density; color=:steelblue, linewidth=2.5, label="Observed")
    lines!(univariate_ax, avg_uni_gen.x, avg_uni_gen.density; color=:tomato, linewidth=2.5, linestyle=:dash, label="Generated")
    axislegend(univariate_ax; position=:rt, framevisible=true, backgroundcolor=:white, labelsize=24)

    # Rows 2-4: Bivariate PDFs for distances 1, 2, 3
    distances = [1, 2, 3]
    for (row_idx, dist) in enumerate(distances)
        biv_row = row_idx + 1  # Rows 2, 3, 4

        # Left column: Observed
        ax_obs = Axis(fig[biv_row, 1];
            xlabel="u(x[i])",
            ylabel=@sprintf("u(x[i+%d])", dist),
            title=@sprintf("Observed: <P(x[i],x[i+%d])>ᵢ", dist),
            titlesize=32,
            xlabelsize=28,
            ylabelsize=28)
        xlims!(ax_obs, biv_xlim)
        ylims!(ax_obs, biv_ylim)

        biv_obs = avg_biv_emp[row_idx]
        heatmap!(ax_obs, biv_obs.x, biv_obs.y, biv_obs.density;
                 colormap=:viridis, colorrange=(0, biv_density_max))

        # Right column: Generated
        ax_gen = Axis(fig[biv_row, 2];
            xlabel="u(x[i])",
            ylabel=@sprintf("u(x[i+%d])", dist),
            title=@sprintf("Generated: <P(x[i],x[i+%d])>ᵢ", dist),
            titlesize=32,
            xlabelsize=28,
            ylabelsize=28)
        xlims!(ax_gen, biv_xlim)
        ylims!(ax_gen, biv_ylim)

        biv_gen = avg_biv_gen[row_idx]
        heatmap!(ax_gen, biv_gen.x, biv_gen.y, biv_gen.density;
                 colormap=:viridis, colorrange=(0, biv_density_max))
    end

    # Add colorbar for bivariate plots
    Colorbar(fig[2:4, 3]; limits=(0, biv_density_max), colormap=:viridis,
             label="Density", ticklabelsize=24, labelsize=26, width=25)

    # Add overall title with metadata
    subtitle = @sprintf("KL = %.4f | modes=%d | stride=%d | clusters=%d",
                       rel_ent, mode_count, stride_tag, clusters_tag)
    Label(fig[0, :], text=subtitle, fontsize=30, font=:bold)

    CairoMakie.save(fig_path, fig)
    @info "Saved diagnostics figure" path=fig_path

    return fig_path, nothing
end

"""
    save_results(path; kwargs...)

Persist all artefacts (latent statistics, diagnostics, metadata) so downstream
notebooks/tests can reproduce the pipeline without re-running costly training.
"""
function save_results(path::AbstractString; kwargs...)
    mkpath(dirname(path))
    h5open(path, "w") do file
        for (key, value) in kwargs
            if value === nothing
                continue
            end
            write(file, String(key), value)
        end
    end
    @info "Saved results" path
end

function create_next_run_dir(base_dir::AbstractString)
    mkpath(base_dir)
    entries = readdir(base_dir)
    pattern = r"^run_(\d+)$"
    max_id = 0
    for entry in entries
        match_entry = match(pattern, entry)
        match_entry === nothing && continue
        run_idx = try
            parse(Int, match_entry.captures[1])
        catch
            continue
        end
        max_id = max(max_id, run_idx)
    end
    run_name = @sprintf("run_%03d", max_id + 1)
    run_dir = joinpath(base_dir, run_name)
    mkpath(run_dir)
    return run_dir
end

function config_to_dict(cfg::KSConfig)
    params = Dict{String,Any}()
    for field in fieldnames(KSConfig)
        value = getfield(cfg, field)
        key = String(field)
        if value isa NamedTuple
            nested = Dict{String,Any}()
            for (k, v) in pairs(value)
                nested[String(k)] = v
            end
            params[key] = nested
        elseif value isa Vector
            params[key] = collect(value)
        elseif value === nothing
            params[key] = ""
        elseif value isa Symbol
            params[key] = String(value)
        else
            params[key] = value
        end
    end
    return params
end

function write_parameters_file(path::AbstractString, cfg::KSConfig; extras::Dict{String,Any}=Dict{String,Any}())
    params = config_to_dict(cfg)
    for (key, value) in extras
        value === nothing && continue
        params[key] = value
    end
    open(path, "w") do io
        TOML.print(io, params)
    end
    @info "Saved parameters" path
    return path
end

function save_training_loss_plot(losses::AbstractVector, run_dir::AbstractString)
    isempty(losses) && return nothing
    loss_fig = Figure(size=(1400, 900), fontsize=32)
    ax = Axis(loss_fig[1, 1]; xlabel="Epoch", ylabel="Training loss", title="Training loss over epochs")
    epochs = collect(1:length(losses))
    losses_float = Float64.(losses)
    lines!(ax, epochs, losses_float; color=:steelblue, linewidth=2.5)
    loss_fig_path = joinpath(run_dir, "training_loss.png")
    CairoMakie.save(loss_fig_path, loss_fig)
    @info "Saved training loss figure" path=loss_fig_path
    return loss_fig_path
end

# --- KS workflow orchestration (formerly `main`) ---------------------------------
# Evaluate sequentially (Shift+Return friendly) or run the whole script at once.

cfg = build_config()
run_dir = create_next_run_dir(KS_RUNS_DIR)
@info "Initialized run" directory=run_dir

# Check threading configuration
n_threads = Threads.nthreads()
n_cores = Sys.CPU_THREADS
if n_threads == 1 && cfg.langevin_ens > 1
    @warn """
    ⚠️  Julia is running with only 1 thread, but you requested $(cfg.langevin_ens) ensemble members.
    Ensemble integration will be VERY SLOW without parallel threading!

    Your system has $(n_cores) CPU threads available.

    TO FIX THIS:

    1. If using VSCode Julia extension:
       - Restart VSCode completely (close and reopen)
       - The .vscode/settings.json has been configured for auto-threading
       - This should give you all $(n_cores) threads

    2. If running from command line:
       julia -t auto --project=. scripts/KS/ks.jl

    3. Or use the provided script:
       ./run_experiment.sh auto --langevin-ens=$(cfg.langevin_ens)

    4. Or set environment variable permanently:
       export JULIA_NUM_THREADS=auto   # Add to ~/.bashrc

    Current: $(n_threads) thread(s) | Available: $(n_cores) cores | Requested: $(cfg.langevin_ens) ensemble members
    """ maxlog=1
elseif n_threads < cfg.langevin_ens
    @info "Threading configuration" available_threads=n_threads available_cores=n_cores ensemble_members=cfg.langevin_ens note="Consider using more threads for better performance"
else
    @info "Threading configuration ✓" threads=n_threads available_cores=n_cores ensemble_members=cfg.langevin_ens status="Optimal"
end

Random.seed!(cfg.seed)

data_raw, delta_t = load_ks_data(KS_DATAFILE; stride=cfg.stride, max_snapshots=cfg.max_snapshots, dataset_key=cfg.dataset_key)
data = Float32.(data_raw)
data_raw = nothing
GC.gc()

coords_centered, retained_indices, mean_field = build_circshifted_modes(data, cfg.n_modes, cfg.train_copies)
data = nothing
GC.gc()

latent_dim = size(coords_centered, 1)
@info "Selected modes" stride=cfg.n_modes latent_dim=latent_dim train_copies=cfg.train_copies

Xn_full, means, stds = normalize_data(coords_centered, cfg.normalize_mode; inplace=true)
coords_centered = nothing

if cfg.train_max > 0 && size(Xn_full, 2) > cfg.train_max
    Xn = @view Xn_full[:, 1:cfg.train_max]
else
    Xn = Xn_full
end
kgmm_sample_count = size(Xn, 2)
@info "Training score model" latent_dim=latent_dim epochs=cfg.n_epochs batch_size=cfg.batch_size train_samples=kgmm_sample_count total_samples=size(Xn_full, 2) train_copies=cfg.train_copies
nn, losses, kgmm_res = train_score_model(Xn, cfg)
final_loss = isempty(losses) ? NaN : losses[end]
@info "Training completed" final_loss
cluster_count = kgmm_res === nothing ? 0 : get(kgmm_res, :Nc, length(get(kgmm_res, :counts, Int[])))
@info "kgmm cluster count" clusters=cluster_count

nn_run_path = joinpath(run_dir, "nn.bson")
BSON.@save nn_run_path nn cfg means stds retained_indices
@info "Saved trained NN" path=nn_run_path

training_loss_fig_path = save_training_loss_plot(losses, run_dir)

samples_n = sample_langevin(latent_dim, nn, cfg)
# kde_Xn = kde(Xn')
# kde_samples_n = kde(samples_n')
# plt1 = heatmap(kde_Xn.x, kde_Xn.y, kde_Xn.density; label="latent coords",
#     title="KDE of normalized latent coordinates", xlabel="reduced coordinate", ylabel="density")
# plt2 = heatmap(kde_samples_n.x, kde_samples_n.y, kde_samples_n.density; label="latent coords",
#     title="KDE of normalized latent coordinates", xlabel="reduced coordinate", ylabel="density")
# plot(plt1, plt2; layout=(1, 2), size=(900, 400))

X_gen_n = Float32.(samples_n)
samples_n = nothing
X_gen = unnormalize_data(X_gen_n, means, stds)

generated_subset = assemble_physical_subset(X_gen, mean_field, retained_indices)
empirical_subset = assemble_physical_subset(unnormalize_data(Xn_full, means, stds), mean_field, retained_indices)

pdf_emp_samples = collect_for_kde(empirical_subset, cfg.kde_max_samples)
pdf_gen_samples = collect_for_kde(generated_subset, cfg.kde_max_samples)
pdf_emp_count = length(pdf_emp_samples)
pdf_gen_count = length(pdf_gen_samples)
@info "PDF sample counts (post-thinning)" empirical=pdf_emp_count generated=pdf_gen_count

obs_pos = 1
observable_idx = retained_indices[obs_pos]
series_emp = vec(empirical_subset[obs_pos, :])
series_gen = vec(generated_subset[obs_pos, :])

emp_tau, eff_emp_temporal = decorrelation_metrics(series_emp)
gen_tau, eff_gen_temporal = decorrelation_metrics(series_gen)
spatial_tau_emp = spatial_decorrelation_length(empirical_subset)
spatial_tau_gen = spatial_decorrelation_length(generated_subset)

kde_emp = estimate_pdf_histogram(pdf_emp_samples)
kde_gen = estimate_pdf_histogram(pdf_gen_samples)
rel_ent = ScoreEstimation.relative_entropy(pdf_emp_samples, pdf_gen_samples; npoints=cfg.rel_entropy_points)
@info "Relative entropy" rel_ent

denom_emp = max(emp_tau * spatial_tau_emp, eps(Float64))
denom_gen = max(gen_tau * spatial_tau_gen, eps(Float64))
eff_emp = pdf_emp_count / denom_emp
eff_gen = pdf_gen_count / denom_gen
kgmm_temporal_effective = kgmm_sample_count / max(emp_tau, eps(Float64))
kgmm_uncorrelated_samples = kgmm_sample_count / denom_emp
@info "Effective sample counts" pdf_empirical=eff_emp pdf_generated=eff_gen kgmm_uncorrelated=kgmm_uncorrelated_samples

# Compute averaged univariate and bivariate PDFs using circular translational invariance
@info "Computing averaged PDFs using circular translational invariance"
range_emp = determine_value_range(empirical_subset)
range_gen = determine_value_range(generated_subset)
value_lo = min(range_emp[1], range_gen[1])
value_hi = max(range_emp[2], range_gen[2])
if value_hi <= value_lo
    span = max(abs(value_lo), 1.0)
    value_lo -= span
    value_hi += span
end
value_range = (value_lo, value_hi)

avg_uni_emp, avg_biv_emp = compute_averaged_pdfs(empirical_subset; value_range=value_range)
avg_uni_gen, avg_biv_gen = compute_averaged_pdfs(generated_subset; value_range=value_range)
@info "Averaged PDFs computed" n_dimensions=size(empirical_subset, 1) n_bivariate_distances=length(avg_biv_emp)

comparison_fig_run_path = joinpath(run_dir, "comparison.png")
analysis_fig_path, modes_fig_path = build_plot(series_emp, series_gen, Xn_full, X_gen,
    kde_emp, kde_gen, avg_uni_emp, avg_uni_gen, avg_biv_emp, avg_biv_gen,
    rel_ent, cfg, cluster_count, delta_t, observable_idx;
    save_path=comparison_fig_run_path)
modes_fig_run_path = ""

run_output_path = joinpath(run_dir, "output.h5")
run_label = basename(run_dir)

result_kwargs = (
    Xn = Xn_full,
    X_gen_n = X_gen_n,
    mean_field = mean_field,
    selected_indices = retained_indices,
    reduced_means = means,
    reduced_stds = stds,
    train_losses = losses,
    final_loss = final_loss,
    sigma = cfg.sigma,
    neurons = collect(Int, cfg.neurons),
    n_epochs = cfg.n_epochs,
    batch_size = cfg.batch_size,
    learning_rate = cfg.lr,
    kgmm_prob = cfg.kgmm_kwargs.prob,
    kgmm_conv_param = cfg.kgmm_kwargs.conv_param,
    kgmm_i_max = cfg.kgmm_kwargs.i_max,
    kgmm_cluster_count = cluster_count,
    kgmm_sample_count = kgmm_sample_count,
    kgmm_temporal_effective_samples = kgmm_temporal_effective,
    kgmm_uncorrelated_samples = kgmm_uncorrelated_samples,
    langevin_boundary = cfg.langevin_boundary,
    langevin_dt = cfg.langevin_dt,
    langevin_steps = cfg.langevin_steps,
    langevin_resolution = cfg.langevin_resolution,
    langevin_ens = cfg.langevin_ens,
    langevin_burnin = cfg.langevin_burnin,
    series_emp = series_emp,
    series_gen = series_gen,
    pdf_emp_sample_count = pdf_emp_count,
    pdf_generated_sample_count = pdf_gen_count,
    empirical_decorrelation_length = emp_tau,
    generated_decorrelation_length = gen_tau,
    empirical_spatial_decorrelation_length = spatial_tau_emp,
    generated_spatial_decorrelation_length = spatial_tau_gen,
    pdf_empirical_temporal_effective_samples = eff_emp_temporal,
    pdf_generated_temporal_effective_samples = eff_gen_temporal,
    pdf_empirical_effective_samples = eff_emp,
    pdf_generated_effective_samples = eff_gen,
    kde_max_samples = cfg.kde_max_samples,
    kde_emp_x = collect(kde_emp.x),
    kde_emp_density = kde_emp.density,
    kde_gen_x = collect(kde_gen.x),
    kde_gen_density = kde_gen.density,
    relative_entropy = rel_ent,
    mode_stride = cfg.n_modes,
    latent_dimension = latent_dim,
    data_stride = cfg.stride,
    max_snapshots = cfg.max_snapshots,
    delta_t = delta_t === nothing ? nothing : [delta_t],
    normalization_mode = String(cfg.normalize_mode),
    train_copies = cfg.train_copies,
    analysis_figure_path = analysis_fig_path,
    analysis_figure_run_path = comparison_fig_run_path,
    comparison_figure_run_path = comparison_fig_run_path,
    nn_path = nn_run_path,
    nn_run_path = nn_run_path,
    modes_figure_path = modes_fig_path === nothing ? "" : modes_fig_path,
    modes_figure_run_path = modes_fig_run_path,
    training_loss_figure_path = training_loss_fig_path === nothing ? "" : training_loss_fig_path,
    run_directory = run_dir,
    run_label = run_label,
    run_output_path = run_output_path,
)

save_results(run_output_path; result_kwargs...)
mkpath(dirname(KS_OUTPUT_H5))
cp(run_output_path, KS_OUTPUT_H5; force=true)

params_path = joinpath(run_dir, "parameters.toml")
extras = Dict{String,Any}(
    "latent_dimension" => latent_dim,
    "cluster_count" => cluster_count,
    "final_loss" => final_loss,
    "run_directory" => run_dir,
    "run_label" => run_label,
    "analysis_figure" => comparison_fig_run_path,
    "modes_figure" => modes_fig_run_path,
    "training_loss_figure" => training_loss_fig_path === nothing ? "" : training_loss_fig_path,
    "nn_run_path" => nn_run_path,
    "output_path" => run_output_path,
    "pdf_empirical_samples" => pdf_emp_count,
    "pdf_generated_samples" => pdf_gen_count,
    "empirical_decorrelation_length" => emp_tau,
    "generated_decorrelation_length" => gen_tau,
    "empirical_spatial_decorrelation_length" => spatial_tau_emp,
    "generated_spatial_decorrelation_length" => spatial_tau_gen,
    "pdf_empirical_temporal_effective_samples" => eff_emp_temporal,
    "pdf_generated_temporal_effective_samples" => eff_gen_temporal,
    "pdf_empirical_effective_samples" => eff_emp,
    "pdf_generated_effective_samples" => eff_gen,
    "kgmm_sample_count" => kgmm_sample_count,
    "kgmm_temporal_effective_samples" => kgmm_temporal_effective,
    "kgmm_uncorrelated_samples" => kgmm_uncorrelated_samples,
    "train_losses" => Float64.(losses),
)
write_parameters_file(params_path, cfg; extras=extras)
