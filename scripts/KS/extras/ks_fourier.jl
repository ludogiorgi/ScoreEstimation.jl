using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# Stand-alone driver that mirrors the reduced 1D workflow in `scripts/reduced/`
# but targets the 32-mode projection of the Kuramoto–Sivashinsky dataset. The
# script performs: data loading, Fourier/PCA compression, score training via
# `ScoreEstimation.train`, Langevin sampling with FastSDE, and
# PDF/relative-entropy diagnostics in physical space.

using FFTW
using FastSDE
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Plots
using Random
using Printf
using ScoreEstimation
using Statistics

const KS_ROOT = @__DIR__
const KS_DATA_DIR = joinpath(KS_ROOT, "data")
const KS_FIG_DIR = joinpath(KS_ROOT, "figures")
const KS_DATAFILE = joinpath(KS_DATA_DIR, "new_ks.hdf5")
const KS_OUTPUT_H5 = joinpath(KS_DATA_DIR, "ks_results.h5")

Plots.gr()

# Configuration container that keeps all runtime parameters together. These
# defaults (stride, σ, network shape, etc.) reflect the heuristics used in the
# reduced 1D scripts but scaled up to 32-dimensional latent coordinates.
struct KSConfig
    stride::Int
    max_snapshots::Int
    modes::Int
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
    sample_keep::Int
    observable_index::Int
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
    stride = parse(Int, get(args, "stride", "1"))                 # Subsampling interval along time axis when reading KS data
    max_snapshots = parse(Int, get(args, "max-snapshots", "0"))   # Hard cap on number of temporal snapshots (0 = no cap)
    modes = parse(Int, get(args, "modes", "6"))                      # Number of retained PCA modes (latent dimensionality)
    seed = parse(Int, get(args, "seed", "2024"))                      # RNG seed for reproducibility (data shuffles, training, sampling)
    sigma = parse(Float64, get(args, "sigma", "0.1"))                # Fixed noise level σ used in ε-training / score scaling
    neurons = [128, 64]
    n_epochs = parse(Int, get(args, "n-epochs", "100"))             # Training epochs for ε-network
    batch_size = parse(Int, get(args, "batch-size", "32"))            # Batch size for Flux DataLoader
    lr = parse(Float64, get(args, "lr", "1e-3"))                      # Learning rate for Adam optimizer
    prob = parse(Float64, get(args, "prob", "1e-6"))                  # kgmm: probability threshold for cluster pruning / weighting
    conv_param = parse(Float64, get(args, "conv-param", "5e-3"))       # kgmm: convergence tolerance
    i_max = parse(Int, get(args, "imax", "120"))                      # kgmm: max iterations
    show_progress = get(args, "show-progress", "false") == "true"     # Verbose kgmm progress display toggle
    kgmm_kwargs = (prob=prob, conv_param=conv_param, i_max=i_max, show_progress=show_progress)  # Aggregated kgmm kwargs
    train_max = parse(Int, get(args, "train-max", "0"))               # Optional cap on number of latent samples used for training (0 = all)
    langevin_boundary = parse(Float64, get(args, "langevin-boundary", "10.0"))  # Hard wall in normalized space (applied per coordinate)
    langevin_dt = parse(Float64, get(args, "langevin-dt", "0.005"))    # Time step for Langevin Euler–Maruyama integration (normalized space)
    langevin_steps = parse(Int, get(args, "langevin-steps", "5000000"))  # Total Langevin steps (including burn-in)
    langevin_resolution = parse(Int, get(args, "langevin-resolution", "20"))  # Recording interval after burn-in (thinning factor)
    langevin_ens = parse(Int, get(args, "langevin-ens", "1"))         # Number of parallel ensemble members during sampling
    langevin_burnin = parse(Int, get(args, "langevin-burnin", "2000"))  # Discard initial steps (thermalization)
    sample_keep = parse(Int, get(args, "sample-keep", "5000000"))        # Optional cap on number of generated latent samples retained for analysis
    observable_index = parse(Int, get(args, "observable-index", "64"))  # Spatial grid index for physical observable PDF comparison
    plot_window = parse(Int, get(args, "plot-window", "1000"))         # Max length of time series segment displayed in subplot 1
    kde_max_samples = parse(Int, get(args, "kde-max-samples", "0"))  # Upper bound on samples fed to KDE (0 = use all)
    rel_entropy_points = parse(Int, get(args, "rel-entropy-points", "2048"))  # Number of quadrature points for relative entropy estimation
    normalize_mode_str = lowercase(get(args, "normalize-mode", "per_dim"))  # Normalization mode: "max" or "per_dim"
    normalize_mode = normalize_mode_str == "max" ? :max : :per_dim
    train_copies = max(1, parse(Int, get(args, "train-copies", "10")))
    dataset_key_str = strip(get(args, "dataset-key", ""))
    dataset_key = isempty(dataset_key_str) ? nothing : dataset_key_str
    return KSConfig(stride, max_snapshots, modes, seed, sigma, neurons, n_epochs, batch_size, lr,
        kgmm_kwargs, train_max, langevin_boundary, langevin_dt, langevin_steps, langevin_resolution,
        langevin_ens, langevin_burnin, sample_keep, observable_index, plot_window,
        kde_max_samples, rel_entropy_points, normalize_mode, train_copies, dataset_key)
end


"""
    load_ks_data(path; stride, max_snapshots, dataset_key=nothing) -> (field, Δt?)

Read the historical KS simulation stored in `ks_old.hdf5`. We lazily stride the
time axis to control memory and optionally cap the number of snapshots so the
downstream PCA/training pipeline remains tractable.
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
                        dims = size(obj)
                        length(dims) == 2 || continue
                        push!(dataset_candidates, String(name))
                    end
                finally
                    close(obj)
                end
            end
            preferred_order = ("u", "timeseries", "timeseries_aligned", "field", "data")
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
    pack_fourier!(dest, coeffs)

Store the reduced real/imag parts of an rFFT result into a purely real buffer.
This matches the layout expected by `fourier_real_matrix` and ensures the PCA
operates on 2·(m-1)+2 real degrees of freedom (DC and Nyquist remain real).
"""
function pack_fourier!(dest::AbstractVector{Float64}, coeffs::AbstractVector{ComplexF64})
    n_freq = length(coeffs)
    dest[1] = real(coeffs[1])
    idx = 2
    for k in 2:(n_freq - 1)
        dest[idx] = real(coeffs[k])
        dest[idx + 1] = imag(coeffs[k])
        idx += 2
    end
    dest[end] = real(coeffs[end])
    return dest
end

"""
    unpack_fourier(vec, n_freq) -> Vector{ComplexF64}

Inverse operation of `pack_fourier!`; recreates the complex spectrum prior to
calling `irfft`. Keeping the serialization logic local avoids aliasing errors
when future contributors tweak the latent dimension.
"""
function unpack_fourier(vec::AbstractVector{<:Real}, n_freq::Int)
    coeffs = Vector{ComplexF64}(undef, n_freq)
    coeffs[1] = Complex(vec[1], 0.0)
    idx = 2
    for k in 2:(n_freq - 1)
        coeffs[k] = Complex(vec[idx], vec[idx + 1])
        idx += 2
    end
    coeffs[n_freq] = Complex(vec[end], 0.0)
    return coeffs
end

"""
    fourier_real_matrix(data) -> Matrix{Float64}

Apply `rfft` snapshot-wise and collapse complex pairs so each column represents
the KS state in the (half) Fourier basis. This is the analogue of the scalar
projection in `reduced_compute.jl`, but we preserve 32 spatial modes here.
"""
function fourier_real_matrix(data::AbstractMatrix)
    nx, nt = size(data)
    n_freq = div(nx, 2) + 1
    output = Matrix{Float64}(undef, nx, nt)
    temp = Vector{Float64}(undef, nx)
    for j in 1:nt
        coeffs = rfft(@view data[:, j])
        pack_fourier!(temp, coeffs)
        output[:, j] = temp
    end
    return output
end

"""
    inverse_fourier_real(vec, nx) -> Vector{Float64}

Recover the physical field by reversing the packing and running `irfft`.
Reattaching the spatial mean happens later in `reconstruct_fields`.
"""
function inverse_fourier_real(vec::AbstractVector{<:Real}, nx::Int)
    n_freq = div(nx, 2) + 1
    coeffs = unpack_fourier(vec, n_freq)
    return real(irfft(coeffs, nx))
end

"""
    compute_pca(data, modes)

Perform an SVD-equivalent PCA on the packed Fourier snapshots and keep the
leading `modes` eigenvectors. Persisted basis/mean mirror the reduced 1D logic
but now capture coherent structures of the full KS attractor.
"""
function compute_pca(data::AbstractMatrix, modes::Int)
    μ = mean(data; dims=2)
    data_centered = data .- μ
    cov = data_centered * transpose(data_centered) / (size(data_centered, 2) - 1)
    evals, evecs = eigen(Symmetric(cov))
    order = sortperm(evals; rev=true)
    basis = evecs[:, order[1:modes]]
    explained = evals[order]
    coords = transpose(basis) * data_centered
    return basis, coords, vec(μ), explained
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
"""
function sample_langevin(dim::Int, nn, cfg::KSConfig)
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
    if total_cols == 0
        return zeros(Float64, dim_out, 0)
    end

    cols_per_ens = max(div(total_cols, cfg.langevin_ens), 1)
    burnin_blocks = ceil(Int, cfg.langevin_burnin / resolution)
    drop_blocks = min(cols_per_ens, burnin_blocks + 1)  # also discard initial condition snapshot
    drop_cols = drop_blocks * cfg.langevin_ens
    keep_start = min(drop_cols, total_cols)
    if keep_start == total_cols
        return zeros(Float64, dim_out, 0)
    end
    return samples_mat[:, keep_start + 1:end]
end

"""
    reconstruct_fields(X, basis, fourier_mean, nx, mean_field)

Project normalized latent samples back to physical space by undoing PCA and the
Fourier packing. This is the KS analogue of rebuilding the scalar SDE signals
in the reduced scripts, but now returns full spatial snapshots.
"""
function reconstruct_fields(X::AbstractMatrix, basis::AbstractMatrix, fourier_mean::AbstractVector, nx::Int, mean_field::AbstractVector)
    fourier_mean_mat = reshape(fourier_mean, :, 1)
    coeffs = basis * X .+ fourier_mean_mat
    n_samples = size(X, 2)
    fields = Matrix{Float64}(undef, nx, n_samples)
    for j in 1:n_samples
        fields[:, j] = inverse_fourier_real(@view(coeffs[:, j]), nx) .+ mean_field
    end
    return fields
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

"""
    select_subset(X, max_count)

Optional subsampling hook so the training set or generated ensemble can be
trimmed deterministically. Useful when sweeping hyperparameters via scripts.
"""
function select_subset(X::AbstractMatrix, max_count::Int)
    if max_count > 0 && size(X, 2) > max_count
        idx = randperm(size(X, 2))[1:max_count]
        return X[:, idx]
    end
    return X
end

"""
    ensure_observable_index(idx, nx) -> Int

Clamp the user-provided observable index into the spatial grid bounds. This
avoids runtime bounds errors when experimenting with different resolutions.
"""
function ensure_observable_index(idx::Int, nx::Int)
    return clamp(idx, 1, nx)
end

"""
    build_plot(series_emp, series_rec, series_gen, X, X_gen, kde_emp, kde_rec, kde_gen, rel_ent, cfg, cluster_count)

Compose diagnostics comparing physical-space observables and latent spreads.
We display raw empirical data, the empirical field reconstructed via PCA
compression, and the samples generated by the learned score model.
"""
function build_plot(series_emp::AbstractVector, series_rec::AbstractVector, series_gen::AbstractVector,
                    X::AbstractMatrix, X_gen::AbstractMatrix,
                    Xn::AbstractMatrix, X_gen_n::AbstractMatrix,
                    kde_emp::KernelDensity.UnivariateKDE, kde_rec::KernelDensity.UnivariateKDE,
                    kde_gen::KernelDensity.UnivariateKDE,
                    rel_ent::Real, cfg::KSConfig, cluster_count::Integer)
    mkpath(KS_FIG_DIR)
    sigma_tag = replace(@sprintf("%0.5f", cfg.sigma), "." => "p")
    modes_tag = cfg.modes
    clusters_tag = max(cluster_count, 0)
    fig_name = @sprintf("ks_analysis_modes%d_sigma%s_clusters%d.png", modes_tag, sigma_tag, clusters_tag)
    fig_modes_name = @sprintf("ks_modes_pdfs_modes%d_sigma%s_clusters%d.png", modes_tag, sigma_tag, clusters_tag)
    fig_path = joinpath(KS_FIG_DIR, fig_name)
    fig_modes_path = joinpath(KS_FIG_DIR, fig_modes_name)
    fig = plot(layout=(3, 1), size=(900, 1200))
    tshow = minimum([cfg.plot_window, length(series_emp), length(series_rec), length(series_gen)])
    plot!(fig[1], 1:tshow, series_emp[1:tshow]; label="empirical", color=:steelblue)
    plot!(fig[1], 1:tshow, series_rec[1:tshow]; label="reconstructed", color=:forestgreen, linestyle=:dot)
    plot!(fig[1], 1:tshow, series_gen[1:tshow]; label="generated", color=:tomato, linestyle=:dash)
    xlabel!(fig[1], "time index")
    ylabel!(fig[1], "u(x*, t)")
    title!(fig[1], "Observable time series (idx=$(cfg.observable_index))")

    std_emp = vec(std(X; dims=2))
    std_gen = vec(std(X_gen; dims=2))
    bar!(fig[2], 1:length(std_emp), std_emp; label="empirical", color=:steelblue, alpha=0.8)
    bar!(fig[2], 1:length(std_gen), std_gen; label="generated", color=:tomato, alpha=0.6)
    xlabel!(fig[2], "Reduced coordinate")
    ylabel!(fig[2], "Std. deviation")
    title!(fig[2], "Reduced coordinate spread")

    plot!(fig[3], kde_emp.x, kde_emp.density; label="empirical PDF", color=:steelblue, lw=2)
    plot!(fig[3], kde_rec.x, kde_rec.density; label="reconstructed PDF", color=:forestgreen, lw=2, linestyle=:dot)
    plot!(fig[3], kde_gen.x, kde_gen.density; label="generated PDF", color=:tomato, lw=2, linestyle=:dash)
    xlabel!(fig[3], "u(x, t)")
    ylabel!(fig[3], "Density")
    title!(fig[3], "PDF comparison (all grid points, KL=$(round(rel_ent; digits=4)))")

    savefig(fig, fig_path)
    @info "Saved diagnostics figure" path=fig_path

    mode_count = size(Xn, 1)
    modes_path_saved = nothing
    if mode_count > 0
        ncols = clamp(mode_count <= 6 ? mode_count : ceil(Int, sqrt(mode_count)), 1, 6)
        nrows = ceil(Int, mode_count / ncols)
        fig_width = max(900, 350 * ncols)
        fig_height = max(400, 280 * nrows)
        fig_modes = plot(layout=(nrows, ncols), size=(fig_width, fig_height))
        for mode_idx in 1:mode_count
            kde_mode_emp = kde(vec(@view Xn[mode_idx, :]))
            kde_mode_gen = kde(vec(@view X_gen_n[mode_idx, :]))
            axis = fig_modes[mode_idx]
            plot!(axis, kde_mode_emp.x, kde_mode_emp.density;
                label="empirical", color=:steelblue, lw=2, legend=:topright)
            plot!(axis, kde_mode_gen.x, kde_mode_gen.density;
                label="generated", color=:tomato, lw=2, linestyle=:dash)
            xlabel!(axis, "Normalized coordinate $(mode_idx)")
            ylabel!(axis, "Density")
        end
        total_axes = nrows * ncols
        if total_axes > mode_count
            dummy_x = [NaN]
            dummy_y = [NaN]
            for extra in (mode_count + 1):total_axes
                plot!(fig_modes[extra], dummy_x, dummy_y; legend=false, framestyle=:none)
            end
        end
        savefig(fig_modes, fig_modes_path)
        @info "Saved per-mode PDF figure" path=fig_modes_path panels=mode_count
        modes_path_saved = fig_modes_path
    end
    return fig_path, modes_path_saved
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

# --- KS workflow orchestration (formerly `main`) ---------------------------------
# Evaluate sequentially (Shift+Return friendly) or run the whole script at once.

cfg = build_config()
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

fourier_matrix = fourier_real_matrix(data_centered)
basis, coords, fourier_mean, explained = compute_pca(fourier_matrix, cfg.modes)

# Normalize using FULL coords to get correct statistics
Xn_full, means, stds = normalize_data(coords, cfg.normalize_mode)

# Truncate AFTER normalization if needed
if cfg.train_max > 0 && size(Xn_full, 2) > cfg.train_max
    Xn = Xn_full[:, 1:cfg.train_max]
else
    Xn = Xn_full
end
@info "Training score model" modes=cfg.modes epochs=cfg.n_epochs batch_size=cfg.batch_size train_samples=size(Xn,2) total_samples=size(coords,2) train_copies=cfg.train_copies
nn, losses, kgmm_res = train_score_model(Xn, cfg)
final_loss = isempty(losses) ? NaN : losses[end]
@info "Training completed" final_loss
cluster_count = kgmm_res === nothing ? 0 : get(kgmm_res, :Nc, length(get(kgmm_res, :counts, Int[])))
@info "kgmm cluster count" clusters=cluster_count

samples_n = sample_langevin(cfg.modes, nn, cfg)

# kde_Xn = kde(Xn')
# kde_samples_n = kde(samples_n')
# plt1 = heatmap(kde_Xn.x, kde_Xn.y, kde_Xn.density; label="latent coords",
#     title="KDE of normalized latent coordinates", xlabel="reduced coordinate", ylabel="density")
# plt2 = heatmap(kde_samples_n.x, kde_samples_n.y, kde_samples_n.density; label="latent coords",
#     title="KDE of normalized latent coordinates", xlabel="reduced coordinate", ylabel="density")
# plot(plt1, plt2; layout=(1, 2), size=(900, 400))

samples_n = select_subset(samples_n, cfg.sample_keep)
X_gen_n = samples_n
X_gen = unnormalize_data(X_gen_n, means, stds)
X_gen
fields_gen = reconstruct_fields(X_gen, basis, fourier_mean, nx, mean_field)
fields_rec = reconstruct_fields(coords, basis, fourier_mean, nx, mean_field)

pdf_emp_samples = collect_for_kde(data, cfg.kde_max_samples)
pdf_rec_samples = collect_for_kde(fields_rec, cfg.kde_max_samples)
pdf_gen_samples = collect_for_kde(fields_gen, cfg.kde_max_samples)
@info "PDF sample counts (post-thinning)" empirical=length(pdf_emp_samples) generated=length(pdf_gen_samples)

observable_idx = ensure_observable_index(cfg.observable_index, nx)
series_emp = vec(data[observable_idx, :])
series_rec = vec(fields_rec[observable_idx, :])
series_gen = vec(fields_gen[observable_idx, :])

kde_emp = kde(pdf_emp_samples)
kde_rec = kde(pdf_rec_samples)
kde_gen = kde(pdf_gen_samples)
rel_ent = ScoreEstimation.relative_entropy(pdf_emp_samples, pdf_gen_samples; npoints=cfg.rel_entropy_points)
@info "Relative entropy" rel_ent

analysis_fig_path, modes_fig_path = build_plot(series_emp, series_rec, series_gen, coords, X_gen,
    Xn_full, X_gen_n,
    kde_emp, kde_rec, kde_gen, rel_ent, cfg, cluster_count)

result_kwargs = (
    Xn = Float32.(Xn_full),
    X_gen_n = Float32.(X_gen_n),
    basis = basis,
    fourier_mean = fourier_mean,
    mean_field = mean_field,
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
    langevin_boundary = cfg.langevin_boundary,
    langevin_dt = cfg.langevin_dt,
    langevin_steps = cfg.langevin_steps,
    langevin_resolution = cfg.langevin_resolution,
    langevin_ens = cfg.langevin_ens,
    langevin_burnin = cfg.langevin_burnin,
    series_emp = series_emp,
    series_reconstructed = series_rec,
    series_gen = series_gen,
    pdf_emp_sample_count = length(pdf_emp_samples),
    pdf_reconstructed_sample_count = length(pdf_rec_samples),
    pdf_generated_sample_count = length(pdf_gen_samples),
    kde_max_samples = cfg.kde_max_samples,
    kde_emp_x = collect(kde_emp.x),
    kde_emp_density = kde_emp.density,
    kde_rec_x = collect(kde_rec.x),
    kde_rec_density = kde_rec.density,
    kde_gen_x = collect(kde_gen.x),
    kde_gen_density = kde_gen.density,
    relative_entropy = rel_ent,
    explained_variance = explained,
    data_stride = cfg.stride,
    max_snapshots = cfg.max_snapshots,
    sample_keep = cfg.sample_keep,
    observable_index = observable_idx,
    delta_t = delta_t === nothing ? nothing : [delta_t],
    normalization_mode = String(cfg.normalize_mode),
    train_copies = cfg.train_copies,
    analysis_figure_path = analysis_fig_path,
    modes_figure_path = modes_fig_path === nothing ? "" : modes_fig_path,
)
save_results(KS_OUTPUT_H5; result_kwargs...)
