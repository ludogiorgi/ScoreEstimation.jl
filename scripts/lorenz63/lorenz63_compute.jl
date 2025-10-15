using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using ScoreEstimation
using FastSDE
using Statistics
using Plots
using Flux
using HDF5
using Random
using KernelDensity
using LinearAlgebra

# Display thread count so end users can confirm multi-threaded execution
@info "Threading status" n_threads=Threads.nthreads()

const L63_ROOT = @__DIR__
const L63_FIGURES_DIR = joinpath(L63_ROOT, "figures")
const L63_DATA_DIR = joinpath(L63_ROOT, "data")
const L63_COMPUTE_FIG = joinpath(L63_FIGURES_DIR, "lorenz63_compute_analysis.png")
const L63_COMPUTE_PDF_FIG = joinpath(L63_FIGURES_DIR, "lorenz63_pdf_comparison.png")
const L63_COMPUTE_H5 = joinpath(L63_DATA_DIR, "lorenz63_compute.h5")

# Lightweight CLI parsing so users can override expensive defaults
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

const CLI_ARGS = parse_args()
cli_int(key, default) = parse(Int, get(CLI_ARGS, key, string(default)))
cli_float(key, default) = parse(Float64, get(CLI_ARGS, key, string(default)))
cli_bool(key; default=false) = get(CLI_ARGS, key, default ? "true" : "false") == "true"

# ---------------- Lorenz-63 stochastic model ----------------
const σ_l63 = 10.0
const ρ_l63 = 28.0
const β_l63 = 8.0 / 3.0
const noise_l63 = 5.0

function lorenz63_drift!(du, u, t)
    du[1] = σ_l63 * (u[2] - u[1])
    du[2] = u[1] * (ρ_l63 - u[3]) - u[2]
    du[3] = u[1] * u[2] - β_l63 * u[3]
    return nothing
end

function lorenz63_diffusion!(du, u, t)
    @inbounds for i in 1:3
        du[i] = noise_l63
    end
    return nothing
end

"""
    normalize_vector(x, mean_vec, std_vec)

Normalize a 3-element vector using per-component statistics.
"""
@inline function normalize_vector(x::AbstractVector, mean_vec::AbstractVector, std_vec::AbstractVector)
    return (x .- mean_vec) ./ std_vec
end

"""
    unnormalize_vector(xn, mean_vec, std_vec)

Unnormalize a 3-element vector using per-component statistics.
"""
@inline function unnormalize_vector(xn::AbstractVector, mean_vec::AbstractVector, std_vec::AbstractVector)
    return xn .* std_vec .+ mean_vec
end

# ---------------- Data generation ----------------
Random.seed!(1234)

dim = 3
dt = cli_float("dt", 0.01)
default_steps = cli_int("n-steps", 10_000_000)
quick_mode = cli_bool("quick"; default=false)
n_steps = quick_mode ? min(default_steps, 200_000) : default_steps
resolution = cli_int("resolution", 10)
raw_ensemble = cli_int("n-ens-data", 1)
n_ens_data = max(raw_ensemble, 1)
boundary_radius = cli_float("boundary", 100.0)
if quick_mode
    @info "Quick mode enabled" n_steps resolution
end
initial_state = [1.0, 1.5, 1.8]

@info "Generating Lorenz-63 trajectories" dt=dt n_steps=n_steps resolution=resolution n_ens=n_ens_data
data = evolve(initial_state, dt, n_steps, lorenz63_drift!, lorenz63_diffusion!;
              timestepper=:euler,
              resolution=resolution,
              sigma_inplace=true,
              n_ens=n_ens_data,
              boundary=(-boundary_radius, boundary_radius),
              flatten=true,
              manage_blas_threads=true)

obs_nn = Array(data)
@info "Trajectory shape" size(obs_nn)

mean_obs = vec(mean(obs_nn; dims=2))
std_obs = vec(std(obs_nn; dims=2))
for i in 1:dim
    std_obs[i] = std_obs[i] > 0 ? std_obs[i] : 1.0
end

obs = (obs_nn .- mean_obs) ./ std_obs

# ---------------- Langevin utilities ----------------
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
        return nothing
    end

    return drift_batched!
end

function generate_langevin_samples(nn, sigma; dt, n_steps, resolution, n_ens, burnin_steps=50_000, boundary=100.0, dim::Int=3)
    drift_batched! = create_batched_drift_nn(nn, sigma)
    init = zeros(Float64, dim)
    lower = -abs(boundary)
    upper = abs(boundary)

    samples = evolve(init, dt, n_steps, drift_batched!, sqrt(2.0);
                     timestepper=:euler,
                     resolution=resolution,
                     n_ens=n_ens,
                     boundary=(lower, upper),
                     flatten=true,
                     manage_blas_threads=true)
    samples_mat = Array(samples)

    snapshots_per_traj = div(n_steps, resolution) + 1
    burnin_snapshots = ceil(Int, burnin_steps / resolution)
    drop_snapshots = min(snapshots_per_traj, burnin_snapshots + 1)
    keep_indices = Int[]
    for ens_idx in 0:(n_ens - 1)
        start_idx = ens_idx * snapshots_per_traj + 1
        keep_start = start_idx + drop_snapshots
        keep_end = (ens_idx + 1) * snapshots_per_traj
        if keep_start <= keep_end
            append!(keep_indices, keep_start:keep_end)
        end
    end
    if isempty(keep_indices)
        @warn "Burn-in removed all Langevin samples" burnin_snapshots snapshots_per_traj
        return zeros(Float64, dim, 0)
    end
    return samples_mat[:, keep_indices]
end

# ---------------- Training ----------------
sigma_value = cli_float("sigma", 0.05)
neurons = [128, 64]
batch_size = cli_int("batch-size", quick_mode ? 16 : 32)
lr = cli_float("lr", 5e-4)
n_epochs = cli_int("n-epochs", quick_mode ? 120 : 1_000)
prob = cli_float("kgmm-prob", 0.001)

Random.seed!(7777)
kgmm_kwargs = (
    prob=prob,
    conv_param=cli_float("kgmm-conv", 0.005),
    i_max=cli_int("kgmm-imax", 200),
    show_progress=true,
)

@info "Training Lorenz-63 score network"
nn, losses, _, _, _, res = ScoreEstimation.train(
    Float32.(obs);
    preprocessing=true,
    σ=sigma_value,
    neurons=neurons,
    n_epochs=n_epochs,
    batch_size=batch_size,
    lr=lr,
    use_gpu=false,
    verbose=true,
    kgmm_kwargs=kgmm_kwargs,
    divergence=false,
)

final_loss = isempty(losses) ? NaN : losses[end]
cluster_count = get(res, :Nc, 0)
@info "Training completed" final_loss cluster_count

# ---------------- Langevin sampling ----------------
dt_gen = cli_float("langevin-dt", 0.005)
n_steps_gen = cli_int("langevin-steps", quick_mode ? 200_000 : 10_000_000)
resolution_gen = cli_int("langevin-resolution", quick_mode ? 10 : 2)
n_ens_gen = cli_int("langevin-ens", 1)
burnin_gen = cli_int("langevin-burnin", quick_mode ? 10_000 : 200_000)
langevin_boundary = cli_float("langevin-boundary", quick_mode ? 30.0 : 100.0)

Random.seed!(2024)
@info "Sampling Langevin trajectories" n_steps=n_steps_gen n_ens=n_ens_gen
samples_nn_norm = generate_langevin_samples(nn, sigma_value;
    dt=dt_gen,
    n_steps=n_steps_gen,
    resolution=resolution_gen,
    n_ens=n_ens_gen,
    burnin_steps=burnin_gen,
    boundary=langevin_boundary,
    dim=dim)

if size(samples_nn_norm, 2) == 0
    error("No Langevin samples survived burn-in. Reduce burn-in or increase integration steps.")
end

samples_nn = samples_nn_norm .* std_obs .+ mean_obs
@info "Generated sample shape" size(samples_nn_norm)

# ---------------- Diagnostics ----------------
rel_ent = ScoreEstimation.relative_entropy(obs, samples_nn_norm; npoints=2048)
@info "Relative entropy (per dimension)" rel_ent

kde_obs_x = kde(obs[1, :])
kde_gen_x = kde(samples_nn_norm[1, :])
kde_obs_y = kde(obs[2, :])
kde_gen_y = kde(samples_nn_norm[2, :])
kde_obs_z = kde(obs[3, :])
kde_gen_z = kde(samples_nn_norm[3, :])

# ---------------- Plotting ----------------
Plots.gr()
mkpath(L63_FIGURES_DIR)

fig = Plots.plot(layout=(4, 1), size=(900, 1600))

plot_len = min(5000, size(obs_nn, 2), size(samples_nn, 2))
time_axis = (0:plot_len-1) .* dt * resolution

Plots.plot!(fig, time_axis, obs_nn[1, 1:plot_len];
    label="observed x",
    xlabel="time",
    ylabel="x",
    title="Lorenz-63 trajectory comparison (x)",
    linewidth=1.5,
    alpha=0.8,
    subplot=1,
    legend=:topright)
Plots.plot!(fig, time_axis, samples_nn[1, 1:plot_len];
    label="NN Langevin x",
    linewidth=1.5,
    linestyle=:dash,
    subplot=1)

Plots.plot!(fig, time_axis, obs_nn[2, 1:plot_len];
    label="observed y",
    xlabel="time",
    ylabel="y",
    title="Lorenz-63 trajectory comparison (y)",
    linewidth=1.5,
    alpha=0.8,
    subplot=2,
    legend=:topright)
Plots.plot!(fig, time_axis, samples_nn[2, 1:plot_len];
    label="NN Langevin y",
    linewidth=1.5,
    linestyle=:dash,
    subplot=2)

Plots.plot!(fig, time_axis, obs_nn[3, 1:plot_len];
    label="observed z",
    xlabel="time",
    ylabel="z",
    title="Lorenz-63 trajectory comparison (z)",
    linewidth=1.5,
    alpha=0.8,
    subplot=3,
    legend=:topright)
Plots.plot!(fig, time_axis, samples_nn[3, 1:plot_len];
    label="NN Langevin z",
    linewidth=1.5,
    linestyle=:dash,
    subplot=3)

Plots.plot!(fig, obs_nn[1, 1:plot_len], obs_nn[3, 1:plot_len];
    seriestype=:scatter,
    markersize=2,
    alpha=0.5,
    label="observed attractor",
    xlabel="x",
    ylabel="z",
    title="Lorenz-63 phase portrait (x-z)",
    subplot=4)
Plots.plot!(fig, samples_nn[1, 1:plot_len], samples_nn[3, 1:plot_len];
    seriestype=:scatter,
    markersize=2,
    alpha=0.5,
    color=:orange,
    label="NN Langevin attractor",
    subplot=4,
    legend=:topright)

display(fig)
Plots.savefig(fig, L63_COMPUTE_FIG)

fig_pdf = Plots.plot(layout=(3, 1), size=(900, 1200))

Plots.plot!(fig_pdf, kde_obs_x.x, kde_obs_x.density;
    label="observed x (norm.)",
    xlabel="x",
    ylabel="density",
    title="Normalized PDF comparison (x)",
    linewidth=2,
    subplot=1)
Plots.plot!(fig_pdf, kde_gen_x.x, kde_gen_x.density;
    label="NN Langevin x",
    linewidth=2,
    linestyle=:dash,
    subplot=1)

Plots.plot!(fig_pdf, kde_obs_y.x, kde_obs_y.density;
    label="observed y (norm.)",
    xlabel="y",
    ylabel="density",
    title="Normalized PDF comparison (y)",
    linewidth=2,
    subplot=2)
Plots.plot!(fig_pdf, kde_gen_y.x, kde_gen_y.density;
    label="NN Langevin y",
    linewidth=2,
    linestyle=:dash,
    subplot=2)

Plots.plot!(fig_pdf, kde_obs_z.x, kde_obs_z.density;
    label="observed z (norm.)",
    xlabel="z",
    ylabel="density",
    title="Normalized PDF comparison (z)",
    linewidth=2,
    subplot=3)
Plots.plot!(fig_pdf, kde_gen_z.x, kde_gen_z.density;
    label="NN Langevin z",
    linewidth=2,
    linestyle=:dash,
    subplot=3)

display(fig_pdf)
Plots.savefig(fig_pdf, L63_COMPUTE_PDF_FIG)

# ---------------- Save diagnostics to HDF5 ----------------
mkpath(L63_DATA_DIR)

h5open(L63_COMPUTE_H5, "w") do file
    write(file, "obs_normalized", Float32.(obs))
    write(file, "obs_physical", Float32.(obs_nn))
    write(file, "samples_normalized", Float32.(samples_nn_norm))
    write(file, "samples_physical", Float32.(samples_nn))
    write(file, "mean_obs", Float64.(mean_obs))
    write(file, "std_obs", Float64.(std_obs))
    write(file, "losses", Float32.(losses))
    write(file, "relative_entropy", Float64.(rel_ent))
    write(file, "kde_obs_x_x", Float32.(kde_obs_x.x))
    write(file, "kde_obs_x_density", Float32.(kde_obs_x.density))
    write(file, "kde_gen_x_x", Float32.(kde_gen_x.x))
    write(file, "kde_gen_x_density", Float32.(kde_gen_x.density))
    write(file, "kde_obs_y_x", Float32.(kde_obs_y.x))
    write(file, "kde_obs_y_density", Float32.(kde_obs_y.density))
    write(file, "kde_gen_y_x", Float32.(kde_gen_y.x))
    write(file, "kde_gen_y_density", Float32.(kde_gen_y.density))
    write(file, "kde_obs_z_x", Float32.(kde_obs_z.x))
    write(file, "kde_obs_z_density", Float32.(kde_obs_z.density))
    write(file, "kde_gen_z_x", Float32.(kde_gen_z.x))
    write(file, "kde_gen_z_density", Float32.(kde_gen_z.density))
    write(file, "trajectory_time", Float64.(time_axis))
    write(file, "trajectory_obs", Float32.(obs_nn[:, 1:plot_len]))
    write(file, "trajectory_gen", Float32.(samples_nn[:, 1:plot_len]))
    write(file, "clusters", Int(cluster_count))
    write(file, "sigma", Float64(sigma_value))
    write(file, "n_epochs", Int(n_epochs))
    write(file, "final_loss", Float64(final_loss))
end

@info "Saved $(L63_COMPUTE_FIG)"
@info "Saved $(L63_COMPUTE_PDF_FIG)"
@info "Saved $(L63_COMPUTE_H5)"

println("\nLorenz-63 analysis summary:")
println("  Clusters           : $(cluster_count)")
println("  Relative entropy   : $(map(x -> round(x, digits=4), rel_ent))")
println("  Final loss         : $(round(final_loss; digits=6))")
