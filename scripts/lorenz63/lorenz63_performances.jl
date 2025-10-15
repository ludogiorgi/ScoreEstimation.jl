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

@info "Threading status" n_threads=Threads.nthreads()

const L63_ROOT = @__DIR__
const L63_FIGURES_DIR = joinpath(L63_ROOT, "figures")
const L63_SCORE_PDFS_DIR = joinpath(L63_FIGURES_DIR, "score_pdfs")
const L63_REL_ENT_FIG = joinpath(L63_FIGURES_DIR, "lorenz63_relative_entropy_vs_time.png")
const L63_DATA_DIR = joinpath(L63_ROOT, "data")
const L63_PERF_H5 = joinpath(L63_DATA_DIR, "lorenz63_performances.h5")

# -----------------------------------------------------------------------------
# CLI utilities
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Lorenz-63 stochastic model
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
Random.seed!(1234)

dim = 3
dt = cli_float("dt", 0.01)
default_steps = cli_int("n-steps", 10_000_000)
quick_mode = cli_bool("quick"; default=false)
n_steps = quick_mode ? min(default_steps, 20_000) : default_steps
resolution = cli_int("resolution", quick_mode ? 20 : 10)
raw_ensemble = cli_int("n-ens-data", 1)
n_ens_data = max(raw_ensemble, 1)
boundary_radius = cli_float("boundary", 100.0)
if quick_mode
    @info "Quick mode enabled" n_steps resolution
end
initial_state = [1.0, 1.5, 1.8]

@info "Generating Lorenz-63 training data" dt=dt n_steps=n_steps resolution=resolution n_ens=n_ens_data
data = evolve(initial_state, dt, n_steps, lorenz63_drift!, lorenz63_diffusion!;
              timestepper=:euler,
              resolution=resolution,
              sigma_inplace=true,
              n_ens=n_ens_data,
              boundary=(-boundary_radius, boundary_radius),
              flatten=true,
              manage_blas_threads=true)

obs_nn = Array(data)
mean_obs = vec(mean(obs_nn; dims=2))
std_obs = vec(std(obs_nn; dims=2))
for i in 1:dim
    std_obs[i] = std_obs[i] > 0 ? std_obs[i] : 1.0
end

obs = (obs_nn .- mean_obs) ./ std_obs
obs_uncorr = obs

if quick_mode
    max_samples = min(size(obs_uncorr, 2), cli_int("quick-max-samples", 6000))
    obs_uncorr = obs_uncorr[:, 1:max_samples]
    @info "Quick mode truncating training samples" n_samples=size(obs_uncorr, 2)
end

# -----------------------------------------------------------------------------
# Langevin helpers
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
function train_without_preprocessing(obs, epochs; sigma, neurons, batch_size, lr, seed,
                                     nn_prev=nothing, cumulative_time=0.0, opt_state_prev=nothing)
    Random.seed!(seed)
    nn = nothing; losses = nothing; opt_state = nothing
    elapsed = @elapsed begin
        nn, losses, _, _, _, _, opt_state = ScoreEstimation.train(
            Float32.(obs);
            preprocessing=false,
            σ=sigma,
            neurons=neurons,
            n_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_gpu=false,
            verbose=false,
            divergence=false,
            nn=nn_prev,
            opt_state=opt_state_prev,
        )
    end
    return nn, losses, cumulative_time + elapsed, opt_state
end

function train_with_preprocessing(obs, epochs, prob;
        sigma, neurons, batch_size, lr, seed,
        conv_param=0.005, i_max=200, show_progress=false)
    Random.seed!(seed)
    kgmm_kwargs = (prob=prob, conv_param=conv_param, i_max=i_max, show_progress=show_progress)
    nn = nothing; losses = nothing; res = nothing
    elapsed = @elapsed begin
        nn, losses, _, _, _, res = ScoreEstimation.train(
            Float32.(obs);
            preprocessing=true,
            σ=sigma,
            neurons=neurons,
            n_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_gpu=false,
            verbose=false,
            kgmm_kwargs=kgmm_kwargs,
            divergence=false,
        )
    end
    return nn, losses, elapsed, res.Nc, prob
end

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
sigma_value = cli_float("sigma", 0.05)
neurons = [128, 64]
batch_size = cli_int("batch-size", quick_mode ? 16 : 32)
lr = cli_float("lr", 5e-4)

default_epochs = quick_mode ? collect(10:10:40) : collect(200:200:1_000)
epochs_schedule = begin
    custom = get(CLI_ARGS, "epochs-schedule", "")
    if isempty(custom)
        default_epochs
    else
        [parse(Int, strip(x)) for x in split(custom, ",") if !isempty(strip(x))]
    end
end
prob_schedule = fill(cli_float("kgmm-prob", 0.001), max(length(epochs_schedule), 1))
preprocessing_epochs = cli_int("preprocessing-epochs", quick_mode ? 120 : 1000)

dt_gen = cli_float("langevin-dt", 0.005)
n_steps_gen = cli_int("langevin-steps", quick_mode ? 200_000 : 10_000_000)
resolution_gen = cli_int("langevin-resolution", quick_mode ? 10 : 2)
n_ens_gen = cli_int("langevin-ens", 1)
burnin_gen = cli_int("langevin-burnin", quick_mode ? 10_000 : 200_000)
langevin_boundary = cli_float("langevin-boundary", quick_mode ? 30.0 : 100.0)

mkpath(L63_FIGURES_DIR)
mkpath(L63_SCORE_PDFS_DIR)

# Warm-up runs to amortize compilation
@info "Warm-up (preprocessing=false)"
train_without_preprocessing(obs_uncorr, 1;
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=9999)

@info "Warm-up (preprocessing=true)"
train_with_preprocessing(obs_uncorr, 1, first(prob_schedule);
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=19_999)

const NoPreResult = NamedTuple{(:epochs,:train_time,:relative_entropy,:final_loss),Tuple{Int,Float64,Vector{Float64},Float64}}
const PreResult   = NamedTuple{(:clusters,:clusters_actual,:prob,:train_time,:relative_entropy,:final_loss),Tuple{Int,Int,Float64,Float64,Vector{Float64},Float64}}
results_no_pre = NoPreResult[]
results_pre    = PreResult[]

# -----------------------------------------------------------------------------
# Sweep without preprocessing
# -----------------------------------------------------------------------------
prev_nn = nothing
prev_opt_state = nothing
cumulative_time_no_pre = 0.0
for (i, epochs) in enumerate(epochs_schedule)
    global prev_nn, prev_opt_state, cumulative_time_no_pre
    seed = 10_000 + i
    prev_nn, losses, cumulative_time_no_pre, prev_opt_state = train_without_preprocessing(obs_uncorr, epochs;
        sigma=sigma_value,
        neurons=neurons,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        nn_prev=prev_nn,
        cumulative_time=cumulative_time_no_pre,
        opt_state_prev=prev_opt_state)
    nn = prev_nn
    t_train = cumulative_time_no_pre
    final_loss = isempty(losses) ? NaN : losses[end]
    Random.seed!(111_111)
    samples_nn = generate_langevin_samples(nn, sigma_value;
        dt=dt_gen,
        n_steps=n_steps_gen,
        resolution=resolution_gen,
        n_ens=n_ens_gen,
        burnin_steps=burnin_gen,
        boundary=langevin_boundary,
        dim=dim)
    rel_ent = ScoreEstimation.relative_entropy(obs_uncorr, samples_nn; npoints=2048)
    push!(results_no_pre, (epochs=epochs, train_time=t_train, relative_entropy=rel_ent, final_loss=final_loss))
    @info "preprocessing=false" epochs=epochs train_time=t_train relative_entropy=rel_ent

    # Save quick diagnostic plot for x-component KDE
    if size(samples_nn, 2) > 0
        kde_obs_x = kde(obs_uncorr[1, :])
        kde_nn_x = kde(samples_nn[1, :])
        fig = Plots.plot(kde_obs_x.x, kde_obs_x.density;
            label="obs (normalized x)", linewidth=2, color=:steelblue,
            xlabel="x_norm", ylabel="density",
            title="No preprocessing: epochs=$(epochs)")
        Plots.plot!(fig, kde_nn_x.x, kde_nn_x.density;
            label="NN sample", linewidth=2, linestyle=:dash, color=:darkorange)
        pdf_path = joinpath(L63_SCORE_PDFS_DIR, "no_pre_epochs_$(lpad(epochs, 3, '0')).png")
        Plots.savefig(fig, pdf_path)
    end
    GC.gc()
end

# -----------------------------------------------------------------------------
# Sweep with preprocessing
# -----------------------------------------------------------------------------
for (i, epochs) in enumerate(epochs_schedule)
    Random.seed!(20_000 + i)
    nn, losses, elapsed, clusters, prob = train_with_preprocessing(obs_uncorr, epochs, prob_schedule[i];
        sigma=sigma_value,
        neurons=neurons,
        batch_size=batch_size,
        lr=lr,
        seed=20_000 + i,
        conv_param=0.005,
        i_max=200,
        show_progress=false)
    final_loss = isempty(losses) ? NaN : losses[end]
    Random.seed!(123_456)
    samples_nn = generate_langevin_samples(nn, sigma_value;
        dt=dt_gen,
        n_steps=n_steps_gen,
        resolution=resolution_gen,
        n_ens=n_ens_gen,
        burnin_steps=burnin_gen,
        boundary=langevin_boundary,
        dim=dim)
    rel_ent = ScoreEstimation.relative_entropy(obs_uncorr, samples_nn; npoints=2048)
    push!(results_pre, (clusters=clusters, clusters_actual=clusters, prob=prob, train_time=elapsed, relative_entropy=rel_ent, final_loss=final_loss))
    @info "preprocessing=true" epochs=epochs clusters=clusters prob=prob train_time=elapsed relative_entropy=rel_ent

    if size(samples_nn, 2) > 0
        kde_obs_x = kde(obs_uncorr[1, :])
        kde_nn_x = kde(samples_nn[1, :])
        fig = Plots.plot(kde_obs_x.x, kde_obs_x.density;
            label="obs (normalized x)", linewidth=2, color=:steelblue,
            xlabel="x_norm", ylabel="density",
            title="KGMM preprocessing: epochs=$(epochs), clusters=$(clusters)")
        Plots.plot!(fig, kde_nn_x.x, kde_nn_x.density;
            label="NN sample", linewidth=2, linestyle=:dash, color=:darkorange)
        pdf_path = joinpath(L63_SCORE_PDFS_DIR, "pre_epochs_$(lpad(epochs, 3, '0')).png")
        Plots.savefig(fig, pdf_path)
    end
    GC.gc()
end

# -----------------------------------------------------------------------------
# Relative entropy summary plot (per dimension)
# -----------------------------------------------------------------------------
dim_rel = dim
train_times_no_pre = [r.train_time for r in results_no_pre]
rel_ent_no_pre = hcat([r.relative_entropy for r in results_no_pre]...)
train_times_pre = [r.train_time for r in results_pre]
rel_ent_pre = hcat([r.relative_entropy for r in results_pre]...)

fig_rel = Plots.plot(layout=(dim_rel, 1), size=(900, 900))
for d in 1:dim_rel
    label_dim = ["x", "y", "z"][d]
    Plots.plot!(fig_rel[d], train_times_no_pre, rel_ent_no_pre[d, :];
        xlabel="Training time (s)",
        ylabel="D_KL",
        xscale=:log10,
        yscale=:log10,
        label="No KGMM",
        linewidth=2,
        marker=:circle,
        color=:firebrick)
    Plots.plot!(fig_rel[d], train_times_pre, rel_ent_pre[d, :];
        label="KGMM",
        linewidth=2,
        marker=:diamond,
        color=:forestgreen)
    Plots.title!(fig_rel[d], "Relative entropy vs. time (dimension $label_dim)")
end
Plots.savefig(fig_rel, L63_REL_ENT_FIG)

# -----------------------------------------------------------------------------
# Persist results
# -----------------------------------------------------------------------------
mkpath(L63_DATA_DIR)

h5open(L63_PERF_H5, "w") do file
    write(file, "epochs_schedule", Int.(epochs_schedule))
    write(file, "prob_schedule", Float64.(collect(prob_schedule)))
    write(file, "results_no_pre_time", Float64.(train_times_no_pre))
    write(file, "results_no_pre_relent", Float64.(rel_ent_no_pre))
    write(file, "results_no_pre_loss", Float64.([r.final_loss for r in results_no_pre]))
    write(file, "results_pre_time", Float64.(train_times_pre))
    write(file, "results_pre_relent", Float64.(rel_ent_pre))
    write(file, "results_pre_loss", Float64.([r.final_loss for r in results_pre]))
    write(file, "results_pre_clusters", Int.([r.clusters for r in results_pre]))
    write(file, "sigma", Float64(sigma_value))
    write(file, "neurons", Int.(neurons))
    write(file, "batch_size", Int(batch_size))
    write(file, "lr", Float64(lr))
    write(file, "langevin_dt", Float64(dt_gen))
    write(file, "langevin_steps", Int(n_steps_gen))
    write(file, "langevin_resolution", Int(resolution_gen))
    write(file, "langevin_ens", Int(n_ens_gen))
    write(file, "langevin_burnin", Int(burnin_gen))
    write(file, "langevin_boundary", Float64(langevin_boundary))
    write(file, "mean_obs", Float64.(mean_obs))
    write(file, "std_obs", Float64.(std_obs))
end

@info "Saved $(L63_REL_ENT_FIG)"
@info "Saved $(L63_PERF_H5)"
