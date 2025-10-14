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
const KS_DATAFILE = joinpath(KS_DATA_DIR, "new_ks.hdf5")
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

    stride = getint("stride", 2)                             # Temporal subsampling interval when reading KS data
    max_snapshots = getint("max-snapshots", 0)               # Hard cap on temporal snapshots (0 = no cap)
    n_modes = getint("n-modes", 8)                           # Spatial stride for mode subsampling (1 = keep all grid points)
    seed = getint("seed", 2024)                              # RNG seed for reproducibility
    sigma = getfloat("sigma", 0.1)                           # Noise level σ used in ε-training / score scaling

    neurons = [128, 64]
    n_epochs = getint("n-epochs", 100)                       # Training epochs for ε-network
    batch_size = getint("batch-size", 64)                    # Batch size for Flux DataLoader
    lr = getfloat("lr", 1e-3)                                # Learning rate for Adam optimizer

    prob = getfloat("prob", 1e-4)                            # kgmm: probability threshold for cluster pruning / weighting
    conv_param = getfloat("conv-param", 5e-3)                # kgmm: convergence tolerance
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
    train_copies = max(1, getint("train-copies", 36))
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
    normalize_data(X, mode) -> (Xn, means, stds)

Standardize latent coordinates according to `mode`. When `mode == :per_dim`
each coordinate is divided by its own standard deviation. When `mode == :max`
all coordinates are scaled by the largest per-dimension standard deviation,
so the most variable component attains unit variance while others remain
subunit. Zero-variance coordinates fall back to `eps(Float64)` to avoid
division by zero.
"""
function normalize_data(X::AbstractMatrix, mode::Symbol)
    means = vec(mean(X; dims=2))
    raw_stds = vec(std(X; dims=2))
    raw_stds = map(x -> x > 0 ? x : eps(Float64), raw_stds)
    stds = if mode == :max
        max_std = maximum(raw_stds)
        scale = max_std > 0 ? max_std : eps(Float64)
        fill(scale, length(raw_stds))
    else
        raw_stds
    end
    Xn = (X .- means) ./ stds
    return Xn, means, stds
end

function unnormalize_data(Xn::AbstractMatrix, means::AbstractVector, stds::AbstractVector)
    means_mat = reshape(means, :, 1)
    stds_mat = reshape(stds, :, 1)
    return (Xn .* stds_mat) .+ means_mat
end

"""
    average_decorrelation_length(series) -> Float64

Estimate the mean decorrelation time (expressed in number of time steps) across
all spatial dimensions of the provided time series matrix. The implementation
uses FFT-based autocorrelation for each dimension and integrates the
autocorrelation until it becomes non-positive, mirroring the classical
integrated autocorrelation time estimator. Returns ``1`` when the input contains
fewer than two snapshots or when a dimension is perfectly decorrelated.
"""
function average_decorrelation_length(series::AbstractMatrix)
    n_dim, n_time = size(series)
    n_dim == 0 && return 0.0
    if n_time < 2
        return 1.0
    end

    padded_len = 1 << ceil(Int, log2(2 * n_time))
    tmp = zeros(Float64, padded_len)
    freq_len = padded_len ÷ 2 + 1
    freq_buffer = Vector{ComplexF64}(undef, freq_len)
    ac_buffer = zeros(Float64, padded_len)
    plan_fwd = plan_rfft(tmp; flags=FFTW.ESTIMATE)
    plan_inv = plan_irfft(freq_buffer, padded_len; flags=FFTW.ESTIMATE)
    inv_padded_len = 1.0 / padded_len
    total_tau = 0.0
    lag_norm = collect(n_time:-1:1)

    for dim in 1:n_dim
        vals = @view series[dim, :]
        μ = mean(vals)
        @inbounds for t in 1:n_time
            tmp[t] = vals[t] - μ
        end
        if n_time < padded_len
            fill!(view(tmp, n_time + 1:padded_len), 0.0)
        end

        mul!(freq_buffer, plan_fwd, tmp)
        @inbounds for k in 1:freq_len
            z = freq_buffer[k]
            freq_buffer[k] = z * conj(z)
        end

        mul!(ac_buffer, plan_inv, freq_buffer)
        @inbounds for k in 1:padded_len
            ac_buffer[k] *= inv_padded_len
        end

        @inbounds for lag in 1:n_time
            ac_buffer[lag] /= lag_norm[lag]
        end

        c0 = ac_buffer[1]
        if !(c0 > 0)
            total_tau += 1.0
            continue
        end

        tau = 1.0
        @inbounds for lag in 2:n_time
            rho = ac_buffer[lag] / c0
            if rho <= 0
                break
            end
            tau += 2.0 * rho
        end
        total_tau += tau
    end

    return total_tau / n_dim
end

"""
    train_score_model(Xn, cfg)

Thin wrapper around `ScoreEstimation.train`. We reuse the same kgmm-assisted
preprocessing branch that powers the reduced experiments and keep the API fully
typed so experiment scripts can introspect losses/cluster counts.
"""
function train_score_model(Xn::AbstractMatrix, cfg::KSConfig)
    obs = Float32.(Xn)
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

function collect_for_kde(mat::AbstractMatrix, max_samples::Int)
    total_entries = length(mat)
    if max_samples <= 0 || total_entries <= max_samples
        return vec(copy(mat))
    end
    samples_per_column = size(mat, 1)
    max_columns = max(1, max_samples ÷ samples_per_column)
    col_stride = max(1, cld(size(mat, 2), max_columns))
    selected = mat[:, 1:col_stride:size(mat, 2)]
    return vec(copy(selected))
end

function collect_for_kde(vec_samples::AbstractVector, max_samples::Int)
    total_entries = length(vec_samples)
    if max_samples <= 0 || total_entries <= max_samples
        return collect(vec_samples)
    end
    step = max(1, cld(total_entries, max_samples))
    out_len = cld(total_entries, step)
    buffer = Vector{eltype(vec_samples)}(undef, out_len)
    idx_out = 1
    @inbounds for idx in 1:step:total_entries
        buffer[idx_out] = vec_samples[idx]
        idx_out += 1
    end
    resize!(buffer, idx_out - 1)
    return buffer
end

function decorrelation_metrics(series::AbstractVector)
    nsamples = length(series)
    if nsamples == 0
        return (0.0, 0.0)
    end
    series_matrix = reshape(series, 1, nsamples)
    tau = average_decorrelation_length(series_matrix)
    tau = tau > 0 ? tau : 1.0
    effective = nsamples / tau
    return (tau, effective)
end

"""
    spatial_decorrelation_length(field; max_samples=4096) -> Float64

Estimate the integrated autocorrelation along the spatial axis of `field`.
The input is assumed to have shape (n_space, n_time) and periodic boundary
conditions in space. To keep the computation tractable for very long time
series, we subsample at most `max_samples` columns when forming the spatial
correlation matrix. Returns ``1`` when fewer than two spatial points or fewer
than two sampled snapshots are available.
"""
function spatial_decorrelation_length(field::AbstractMatrix; max_samples::Int=4096)
    n_space, n_time = size(field)
    (n_space <= 1 || n_time == 0) && return 1.0

    step = max(1, n_time ÷ max_samples)
    sampled_indices = 1:step:n_time
    sample_count = length(sampled_indices)
    sample_count <= 1 && return 1.0

    means = zeros(Float64, n_space)
    for idx in sampled_indices
        col = @view field[:, idx]
        @inbounds for i in 1:n_space
            means[i] += col[i]
        end
    end
    @. means /= sample_count

    cov = zeros(Float64, n_space, n_space)
    for idx in sampled_indices
        col = @view field[:, idx]
        @inbounds for i in 1:n_space
            xi = col[i] - means[i]
            cov[i, i] += xi * xi
            for j in i+1:n_space
                xj = col[j] - means[j]
                cov[i, j] += xi * xj
            end
        end
    end

    scale = 1.0 / (sample_count - 1)
    for i in 1:n_space
        cov[i, i] *= scale
        for j in i+1:n_space
            cov[i, j] *= scale
            cov[j, i] = cov[i, j]
        end
    end

    vars = similar(means)
    for i in 1:n_space
        v = cov[i, i]
        if !(v > 0)
            v = eps(Float64)
            cov[i, i] = v
        end
        vars[i] = v
    end

    for i in 1:n_space
        cov[i, i] = 1.0
        for j in i+1:n_space
            denom = sqrt(vars[i] * vars[j])
            val = denom > 0 ? cov[i, j] / denom : 0.0
            val = clamp(val, -1.0, 1.0)
            cov[i, j] = val
            cov[j, i] = val
        end
    end

    tau = 1.0
    for lag in 1:(n_space - 1)
        total = 0.0
        @inbounds for i in 1:n_space
            j = i + lag
            j > n_space && (j -= n_space)
            total += cov[i, j]
        end
        avg_corr = total / n_space
        if !isfinite(avg_corr) || avg_corr <= 0
            break
        end
        tau += 2.0 * avg_corr
    end
    return tau > 0 ? tau : 1.0
end

"""
    PDFEstimate

Simple structure to hold univariate PDF estimation results.
Compatible with the plotting code that previously used KernelDensity.jl.
"""
struct PDFEstimate
    x::Vector{Float64}
    density::Vector{Float64}
end

"""
    BivariatePDFEstimate

Structure to hold bivariate PDF estimation results.
"""
struct BivariatePDFEstimate
    x::Vector{Float64}
    y::Vector{Float64}
    density::Matrix{Float64}
end

"""
    estimate_pdf_histogram(data; nbins=nothing, bandwidth=nothing) -> PDFEstimate

Estimate a univariate PDF using histogram binning with Gaussian smoothing.
This replaces KernelDensity.jl's kde() function.

# Arguments
- `data`: Vector of samples
- `nbins`: Number of bins (default: sqrt(n)/2 clamped to 50-200)
- `bandwidth`: Smoothing bandwidth in data units (default: adaptive based on data range)

# Returns
- `PDFEstimate`: Structure containing grid points and density values
"""
function estimate_pdf_histogram(data::AbstractVector; nbins::Union{Nothing,Int}=nothing, bandwidth::Union{Nothing,Float64}=nothing)
    n = length(data)
    n == 0 && return PDFEstimate(Float64[], Float64[])

    # Determine number of bins
    if nbins === nothing
        nbins = clamp(Int(round(sqrt(n) / 2)), 50, 200)
    end

    # Get data range
    data_min, data_max = extrema(data)
    if data_min == data_max
        # Degenerate case: all values are the same
        return PDFEstimate([data_min], [Inf])
    end

    # Create histogram
    bin_edges = range(data_min, data_max; length=nbins + 1)
    bin_width = (data_max - data_min) / nbins
    counts = zeros(Float64, nbins)

    # Fill histogram
    for val in data
        if isfinite(val)
            bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins)
            counts[bin_idx] += 1
        end
    end

    # Normalize to get density
    counts ./= (n * bin_width)

    # Apply Gaussian smoothing if requested
    if bandwidth === nothing
        # Adaptive bandwidth: use a fraction of the data range
        bandwidth = (data_max - data_min) / 30.0
    end

    if bandwidth > 0
        # Smooth the histogram with a Gaussian kernel
        sigma_bins = bandwidth / bin_width  # Convert to bin units
        kernel_radius = ceil(Int, 3 * sigma_bins)  # 3-sigma cutoff

        if kernel_radius > 0
            smoothed = zeros(Float64, nbins)
            for i in 1:nbins
                weight_sum = 0.0
                for j in max(1, i - kernel_radius):min(nbins, i + kernel_radius)
                    dist = (i - j) * bin_width
                    weight = exp(-0.5 * (dist / bandwidth)^2)
                    smoothed[i] += counts[j] * weight
                    weight_sum += weight
                end
                smoothed[i] /= weight_sum
            end
            counts = smoothed
        end
    end

    # Create grid points at bin centers
    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins))

    return PDFEstimate(x_centers, counts)
end

"""
    estimate_bivariate_pdf_histogram(data_x, data_y; nbins=nothing, x_range=nothing, y_range=nothing) -> BivariatePDFEstimate

Estimate a bivariate PDF using 2D histogram binning.

# Arguments
- `data_x`: Vector of x samples
- `data_y`: Vector of y samples
- `nbins`: Number of bins per dimension (default: sqrt(n)/4 clamped to 30-100)
- `x_range`: Tuple (x_min, x_max) to define x-axis range (default: data extrema)
- `y_range`: Tuple (y_min, y_max) to define y-axis range (default: data extrema)

# Returns
- `BivariatePDFEstimate`: Structure containing grid points and 2D density
"""
function estimate_bivariate_pdf_histogram(data_x::AbstractVector, data_y::AbstractVector;
                                         nbins::Union{Nothing,Int}=nothing,
                                         x_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                                         y_range::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    n = length(data_x)
    @assert length(data_y) == n "data_x and data_y must have the same length"
    n == 0 && return BivariatePDFEstimate(Float64[], Float64[], zeros(Float64, 0, 0))

    # Determine number of bins
    if nbins === nothing
        nbins = clamp(Int(round(sqrt(n) / 4)), 30, 100)
    end

    # Get data ranges
    if x_range === nothing
        x_min, x_max = extrema(data_x)
    else
        x_min, x_max = x_range
    end

    if y_range === nothing
        y_min, y_max = extrema(data_y)
    else
        y_min, y_max = y_range
    end

    if x_min == x_max || y_min == y_max
        return BivariatePDFEstimate([x_min], [y_min], zeros(Float64, 1, 1))
    end

    # Create histogram
    x_edges = range(x_min, x_max; length=nbins + 1)
    y_edges = range(y_min, y_max; length=nbins + 1)
    x_width = (x_max - x_min) / nbins
    y_width = (y_max - y_min) / nbins
    density = zeros(Float64, nbins, nbins)

    # Fill histogram
    valid_count = 0
    for i in 1:n
        xv = data_x[i]
        yv = data_y[i]
        if isfinite(xv) && isfinite(yv) && x_min <= xv <= x_max && y_min <= yv <= y_max
            xi = clamp(searchsortedlast(x_edges, xv), 1, nbins)
            yi = clamp(searchsortedlast(y_edges, yv), 1, nbins)
            density[yi, xi] += 1
            valid_count += 1
        end
    end

    # Normalize to get density
    area = x_width * y_width
    if valid_count > 0 && area > 0
        density ./= (valid_count * area)
    end

    # Create grid points at bin centers
    x_centers = collect(range(x_min + x_width/2, x_max - x_width/2; length=nbins))
    y_centers = collect(range(y_min + y_width/2, y_max - y_width/2; length=nbins))

    return BivariatePDFEstimate(x_centers, y_centers, density)
end

"""
    determine_value_range(data; clip_fraction=0.001, max_samples=1_000_000)

Estimate a robust value range for histogramming by clipping extreme outliers.
Returns a tuple `(lo, hi)` expanded by a small safety margin.
"""
function determine_value_range(data::AbstractMatrix;
                               clip_fraction::Float64=0.001,
                               max_samples::Int=1_000_000)
    samples = collect_for_kde(data, max_samples)
    samples64 = Float64.(samples)
    filter!(isfinite, samples64)
    isempty(samples64) && return (-1.0, 1.0)
    α = clamp(clip_fraction, 0.0, 0.5)
    lo = quantile(samples64, α)
    hi = quantile(samples64, 1 - α)
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        lo, hi = extrema(samples64)
    end
    if !isfinite(lo) || !isfinite(hi)
        return (-1.0, 1.0)
    end
    if hi <= lo
        padding = max(abs(lo), 1.0) * 1e-3 + 1e-6
        lo -= padding
        hi += padding
    end
    span = hi - lo
    pad = span == 0 ? max(abs(lo), 1.0) * 0.01 : span * 0.02
    return (lo - pad, hi + pad)
end

"""
    compute_averaged_pdfs(data::AbstractMatrix; value_range=nothing)
        -> (PDFEstimate, Vector{BivariatePDFEstimate})

Compute averaged univariate and bivariate PDFs using circular translational
invariance. Optionally enforces a predefined `(lo, hi)` range, which helps keep
the histogram support consistent across datasets.
"""
function compute_averaged_pdfs(data::AbstractMatrix;
                               value_range::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    n_dims, n_times = size(data)

    total_samples = n_dims * n_times
    nbins_uni = clamp(Int(round(sqrt(total_samples) / 2)), 50, 200)

    data_min = 0.0
    data_max = 0.0
    if value_range === nothing
        all_values = Float64[]
        sizehint!(all_values, total_samples)
        for i in 1:n_dims
            append!(all_values, Float64.(data[i, :]))
        end
        data_min, data_max = extrema(all_values)
    else
        data_min, data_max = value_range
    end
    if !isfinite(data_min) || !isfinite(data_max) || data_max <= data_min
        finite_vals = Float64[]
        for i in 1:n_dims
            vals = Float64.(data[i, :])
            for v in vals
                isfinite(v) && push!(finite_vals, v)
            end
        end
        if isempty(finite_vals)
            data_min, data_max = -1.0, 1.0
        else
            data_min, data_max = extrema(finite_vals)
            if data_max <= data_min
                spread = max(abs(data_min), 1.0)
                data_min -= spread
                data_max += spread
            end
        end
    end

    bin_edges = range(data_min, data_max; length=nbins_uni + 1)
    bin_width = (data_max - data_min) / nbins_uni
    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins_uni))

    avg_density = zeros(Float64, nbins_uni)
    for i in 1:n_dims
        samples = vec(Float64.(data[i, :]))
        counts = zeros(Float64, nbins_uni)
        valid_len = 0
        for val in samples
            if isfinite(val) && data_min <= val <= data_max
                bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins_uni)
                counts[bin_idx] += 1
                valid_len += 1
            end
        end
        if valid_len > 0
            counts ./= (valid_len * bin_width)
            avg_density .+= counts
        end
    end
    avg_density ./= n_dims

    avg_univariate = PDFEstimate(x_centers, avg_density)

    distances = [1, 2, 3]
    avg_bivariates = BivariatePDFEstimate[]

    nbins_biv = clamp(Int(round(sqrt(n_times) / 4)), 30, 100)
    x_biv_edges = range(data_min, data_max; length=nbins_biv + 1)
    y_biv_edges = x_biv_edges
    x_biv_width = (data_max - data_min) / nbins_biv
    y_biv_width = x_biv_width
    x_biv_centers = collect(range(data_min + x_biv_width/2, data_max - x_biv_width/2; length=nbins_biv))
    y_biv_centers = x_biv_centers

    for dist in distances
        avg_biv_density = zeros(Float64, nbins_biv, nbins_biv)

        for i in 1:n_dims
            j = mod1(i + dist, n_dims)  # Circular indexing
            samples_x = vec(Float64.(data[i, :]))
            samples_y = vec(Float64.(data[j, :]))

            density = zeros(Float64, nbins_biv, nbins_biv)
            valid_count = 0

            for t in 1:n_times
                xv = samples_x[t]
                yv = samples_y[t]
                if isfinite(xv) && isfinite(yv) &&
                   data_min <= xv <= data_max &&
                   data_min <= yv <= data_max
                    xi = clamp(searchsortedlast(x_biv_edges, xv), 1, nbins_biv)
                    yi = clamp(searchsortedlast(y_biv_edges, yv), 1, nbins_biv)
                    density[yi, xi] += 1
                    valid_count += 1
                end
            end

            area = x_biv_width * y_biv_width
            if valid_count > 0 && area > 0
                density ./= (valid_count * area)
            end

            avg_biv_density .+= density
        end

        avg_biv_density ./= n_dims
        push!(avg_bivariates, BivariatePDFEstimate(x_biv_centers, y_biv_centers, avg_biv_density))
    end

    return avg_univariate, avg_bivariates
end

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

data, delta_t = load_ks_data(KS_DATAFILE; stride=cfg.stride, max_snapshots=cfg.max_snapshots, dataset_key=cfg.dataset_key)
std(data, dims=2)

# Stack circshifts of `data` side-by-side (along columns)
function stack_circshifts(data, copies::Integer)
    copies = max(copies, 1)
    copies == 1 && return data
    nrows, ncols = size(data)
    out = similar(data, nrows, ncols * copies)
    @inbounds for i in 0:(copies - 1)
        # shift down by i rows (wrap-around), no column shift
        @views out[:, i * ncols + 1:(i + 1) * ncols] = circshift(data, (i, 0))
    end
    return out
end

joined_data = stack_circshifts(data, cfg.train_copies)
data = joined_data


# kde_xx = kde(data[[1,3], :]')
# heatmap(kde_xx.x, kde_xx.y, kde_xx.density; label="latent coords",
#    title="KDE of first two spatial points", xlabel="u(x=1)", ylabel="u(x=2)")

nx, n_snapshots = size(data)
# decorrelation_steps = average_decorrelation_length(data)
# @info "Average decorrelation length" steps=decorrelation_steps
mean_field = vec(mean(data; dims=2))
data_centered = data .- mean_field

coords, retained_indices = select_modes(data_centered, cfg.n_modes)
latent_dim = size(coords, 1)
@info "Selected modes" stride=cfg.n_modes latent_dim=latent_dim

# Normalize using FULL coords to get correct statistics
Xn_full, means, stds = normalize_data(coords, cfg.normalize_mode)

# Truncate AFTER normalization if needed
if cfg.train_max > 0 && size(Xn_full, 2) > cfg.train_max
    Xn = Xn_full[:, 1:cfg.train_max]
else
    Xn = Xn_full
end
kgmm_sample_count = size(Xn, 2)
@info "Training score model" latent_dim=latent_dim epochs=cfg.n_epochs batch_size=cfg.batch_size train_samples=kgmm_sample_count total_samples=size(coords,2) train_copies=cfg.train_copies
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

X_gen_n = samples_n
X_gen = unnormalize_data(X_gen_n, means, stds)

generated_subset = assemble_physical_subset(X_gen, mean_field, retained_indices)
empirical_subset = data[retained_indices, :]

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
analysis_fig_path, modes_fig_path = build_plot(series_emp, series_gen, coords, X_gen,
    kde_emp, kde_gen, avg_uni_emp, avg_uni_gen, avg_biv_emp, avg_biv_gen,
    rel_ent, cfg, cluster_count, delta_t, observable_idx;
    save_path=comparison_fig_run_path)
modes_fig_run_path = ""

run_output_path = joinpath(run_dir, "output.h5")
run_label = basename(run_dir)

result_kwargs = (
    Xn = Float32.(Xn_full),
    X_gen_n = Float32.(X_gen_n),
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
