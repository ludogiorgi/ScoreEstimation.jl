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

# Show thread count to ensure we are using all available threads when run with `-t auto` or JULIA_NUM_THREADS
@info "Threading status" n_threads=Threads.nthreads()
const REDUCED_ROOT = @__DIR__
const REDUCED_FIGURES_DIR = joinpath(REDUCED_ROOT, "figures")
const REDUCED_SCORE_PDFS_DIR = joinpath(REDUCED_FIGURES_DIR, "score_pdfs")
const REDUCED_REL_ENT_FIG = joinpath(REDUCED_FIGURES_DIR, "reduced_relative_entropy_vs_time.png")
const REDUCED_DATA_DIR = joinpath(REDUCED_ROOT, "data")
const REDUCED_PERF_H5 = joinpath(REDUCED_DATA_DIR, "reduced_performances.h5")

# ---------------- Reduced one-dimensional model ----------------
const reduced_params = (
    L11 = -2.0,
    L12 = 0.2,
    L13 = 0.1,
    g2  = 0.6,
    g3  = 0.4,
    s2  = 1.2,
    s3  = 0.8,
    I   = 1.0,
    eps = 0.1,
)

function reduced_coefficients(p)
    a = p.L11 + p.eps * ((p.I^2 * p.s2^2) / (2 * p.g2^2) - (p.L12^2) / p.g2 - (p.L13^2) / p.g3)
    b = -2 * (p.L12 * p.I) / p.g2 * p.eps
    c = (p.I^2) / p.g2 * p.eps
    B = -(p.I * p.s2) / p.g2 * sqrt(p.eps)
    A = -(p.L12 * B) / p.I
    F_tilde = (A * B) / 2
    s = (p.L13 * p.s3) / p.g3 * sqrt(p.eps)
    return (a=a, b=b, c=c, A=A, B=B, F_tilde=F_tilde, s=s)
end

const coeffs = reduced_coefficients(reduced_params)

function drift_reduced!(du, u, t)
    du[1] = -coeffs.F_tilde + coeffs.a * u[1] + coeffs.b * u[1]^2 - coeffs.c * u[1]^3
    return nothing
end

function diffusion_reduced!(du, u, t)
    du[1] = sqrt((coeffs.A - coeffs.B * u[1])^2 + coeffs.s^2)
    return nothing
end

"""
    score_true_physical(x)

Analytic score s(x) = ∂x log p(x) for the 1D SDE
    dX = f(X) dt + g(X) dW,
with
    f(x) = -F̃ + a x + b x^2 - c x^3,
    g(x)^2 = ((A - B x)^2 + s^2).

Zero-flux stationary solution satisfies: f p - (1/2)∂x(g^2 p) = 0 ⇒
    s(x) = ∂x log p(x) = (2 f(x) - ∂x g(x)^2) / g(x)^2.

Here ∂x g^2 = -B (A - B x), so
    s(x) = (2 f(x) + 2 B (A - B x)) / ( ((A - B x)^2 + s^2) )
         = 2 (f(x) + B (A - B x)) / ((A - B x)^2 + s^2).
"""
function score_true_physical(x::Real)
    f = -coeffs.F_tilde + coeffs.a * x + coeffs.b * x^2 - coeffs.c * x^3
    g2 = (coeffs.A - coeffs.B * x)^2 + coeffs.s^2
    dg2 = -2*coeffs.B * (coeffs.A - coeffs.B * x)
    return (2*f - dg2) / g2
end

score_true_normalized(x::Real, mean_obs::Real, std_obs::Real) =
    std_obs * score_true_physical(x * std_obs + mean_obs)

# ---------------- Data generation ----------------
Random.seed!(1234)

dim = 1
dt = 0.01
n_steps = 50_000
resolution = 100
initial_state = zeros(dim)

obs_nn = evolve(initial_state, dt, n_steps, drift_reduced!, diffusion_reduced!;
                timestepper=:euler, resolution=resolution, sigma_inplace=true, n_ens=100, boundary=(-10,10))

@info "Trajectory shape" size(obs_nn)

mean_obs = mean(obs_nn)
std_obs = std(obs_nn)
obs = (obs_nn .- mean_obs) ./ std_obs
obs_uncorr = obs

score_true_norm_scalar(x::Real) = score_true_normalized(x, mean_obs, std_obs)
# ---------------- Langevin utilities ----------------
"""
    generate_langevin_samples(score_scalar_fn; dt, n_steps, resolution, n_ens, boundary=(-5,5), nn=nothing, sigma=nothing)

Sample using FastSDE.evolve with drift = score and diffusion = sqrt(2) (normalized space).
If `nn` and `sigma` are provided, uses batched drift for faster ensemble integration with neural networks.
"""
function generate_langevin_samples(score_scalar_fn; dt, n_steps, resolution, n_ens, boundary=(-10,10), nn=nothing, sigma=nothing, burnin=1000)
    # Decide if we should use batched mode (only makes sense if an NN is provided AND n_ens > 1)
    use_batched = !isnothing(nn) && !isnothing(sigma)
    if use_batched && n_ens == 1
        # Single ensemble member gives no batching benefit; also avoids arity issues.
        use_batched = false
    end

    if use_batched
        # Batched drift for neural network
        drift_batched! = create_batched_drift_nn(nn, sigma)

        # For batched mode, pass sigma as a scalar Number (FastSDE batched path requirement)
        sigma_constant = sqrt(2.0)
        samples = evolve([0.0], dt, n_steps, drift_batched!, sigma_constant;
                          timestepper=:euler,
                          resolution=resolution,
                          n_ens=n_ens,
                          boundary=boundary,
                          flatten=true,
                          manage_blas_threads=true)
        # Manual burn-in removal (mirror ks.jl logic)
        samples_mat = Array(samples)
        snapshots_per_traj = div(n_steps, resolution) + 1
        burnin_snapshots = ceil(Int, burnin / resolution)
        drop_snapshots_per_traj = min(snapshots_per_traj, burnin_snapshots + 1)
        keep_indices = Int[]
        for ens_idx in 0:(n_ens - 1)
            traj_start = ens_idx * snapshots_per_traj + 1
            keep_start = traj_start + drop_snapshots_per_traj
            keep_end = (ens_idx + 1) * snapshots_per_traj
            if keep_start <= keep_end
                append!(keep_indices, keep_start:keep_end)
            end
        end
        if isempty(keep_indices)
            @warn "All samples removed during burn-in" burnin_snapshots drop_snapshots_per_traj snapshots_per_traj
            return Float64[]
        end
        return vec(samples_mat[:, keep_indices])
    else
        # Non-batched mode for scalar functions
        function drift!(du, u, t)
            du[1] = score_scalar_fn(u[1])
            return nothing
        end
        function unit_diffusion!(du, u, t)
            du[1] = sqrt(2.0)
            return nothing
        end
        samples = evolve([0.0], dt, n_steps, drift!, unit_diffusion!;
                         timestepper=:euler,
                         resolution=resolution,
                         sigma_inplace=true,
                         n_ens=n_ens,
                         boundary=boundary,
                         flatten=true,
                         manage_blas_threads=false)
        samples_mat = Array(samples)
        snapshots_per_traj = div(n_steps, resolution) + 1
        burnin_snapshots = ceil(Int, burnin / resolution)
        drop_snapshots_per_traj = min(snapshots_per_traj, burnin_snapshots + 1)
        keep_indices = Int[]
        for ens_idx in 0:(n_ens - 1)
            traj_start = ens_idx * snapshots_per_traj + 1
            keep_start = traj_start + drop_snapshots_per_traj
            keep_end = (ens_idx + 1) * snapshots_per_traj
            if keep_start <= keep_end
                append!(keep_indices, keep_start:keep_end)
            end
        end
        if isempty(keep_indices)
            @warn "All samples removed during burn-in" burnin_snapshots drop_snapshots_per_traj snapshots_per_traj
            return Float64[]
        end
        return vec(samples_mat[:, keep_indices])
    end
end

Random.seed!(42)
dt_gen = 0.0025
n_steps_gen = 100_000
resolution_gen = 200
n_ens_gen = 100  # increased for more stable KL estimates

# Generate from the original SDE in physical space
true_samples = generate_langevin_samples(score_true_norm_scalar;
    dt=dt_gen,
    n_steps=n_steps_gen,
    resolution=resolution_gen,
    n_ens=n_ens_gen,
    boundary=(-10, 10))

kde_true = kde(true_samples)

const REL_ENT_POINTS = 2048

# ---------------- Helpers ----------------
function score_from_nn_scalar(nn, sigma)
    buf = zeros(Float32, 1, 1)
    function inner(x::Real)
        buf[1] = Float32(x)
        y = nn(buf)
        return -Float64(y[1]) / Float64(sigma)
    end
    return inner
end

# Batched drift function for neural network score
function create_batched_drift_nn(nn, sigma::Real)
    inv_sigma = 1.0 / Float64(sigma)
    # Per-thread buffer store to avoid allocations and ensure thread-safety
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
        # Support both vector (single trajectory) and matrix (batched) inputs
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

function train_without_preprocessing(obs, epochs; sigma, neurons, batch_size, lr, seed,
                                     nn_prev=nothing, cumulative_time=0.0, opt_state_prev=nothing)
    # Only set seed if starting fresh training (no previous network)
    if isnothing(nn_prev)
        Random.seed!(seed)
    end
    nn = nothing; losses = nothing; opt_state = nothing
    elapsed = @elapsed begin
        nn, losses, _, _, _, _, opt_state = ScoreEstimation.train(
            obs;
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
        conv_param=1e-4, i_max=120, show_progress=false)
    Random.seed!(seed)
    kgmm_kwargs = (prob=prob, conv_param=conv_param, i_max=i_max, show_progress=show_progress)
    nn = nothing; losses = nothing; res = nothing
    elapsed = @elapsed begin
        nn, losses, _, _, _, res = ScoreEstimation.train(
            obs;
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

# ---------------- Experiment configuration ----------------
sigma_value = 0.05  # Increased from 0.05 for more stable training
neurons = [100, 50]
batch_size_no_pre = 64  # Larger batch size for preprocessing=false for more stable gradients
batch_size_pre = 8    # Smaller batch size for preprocessing=true (with KGMM)
lr = 1e-3

epochs_schedule = [10:10:100...]
prob_schedule = [0.01:-0.001:0.002...]
preprocessing_epochs = 200

Plots.gr()
mkpath(REDUCED_FIGURES_DIR)
mkpath(REDUCED_SCORE_PDFS_DIR)

# Warm-up runs to remove compilation bias from first measurements
@info "Warm-up (preprocessing=false)"
train_without_preprocessing(obs_uncorr, 1;
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size_no_pre,
    lr=lr,
    seed=9999)

@info "Warm-up (preprocessing=true)"
train_with_preprocessing(obs_uncorr, 1, prob_schedule[1];
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size_pre,
    lr=lr,
    seed=19_999)

# ---------------- Sweeps ----------------
# Result containers (dynamic). We use concrete NamedTuple types so downstream
# comprehensions produce concretely typed arrays (avoids Vector{Any} and HDF5 write errors).
const NoPreResult = NamedTuple{(:epochs,:train_time,:relative_entropy,:final_loss),Tuple{Int,Float64,Float64,Float64}}
const PreResult   = NamedTuple{(:clusters,:clusters_actual,:prob,:train_time,:relative_entropy,:final_loss),Tuple{Int,Int,Float64,Float64,Float64,Float64}}
results_no_pre = NoPreResult[]
results_pre    = PreResult[]

# ---------------- Efficient training for preprocessing=false ----------------
# Train once for maximum epochs and save checkpoints at each delta_epoch
@info "Training without preprocessing for $(maximum(epochs_schedule)) epochs with checkpoints"
delta_epoch = epochs_schedule[2] - epochs_schedule[1]  # Assuming uniform spacing
max_epochs = maximum(epochs_schedule)
seed = 10_000

# Train incrementally, saving checkpoints at specified intervals
nn_current = nothing
opt_state = nothing
cumulative_time = 0.0
all_losses = Float64[]

for (i, target_epochs) in enumerate(epochs_schedule)
    prev_epochs = i > 1 ? epochs_schedule[i-1] : 0
    epochs_to_train = target_epochs - prev_epochs
    
    # Train for the next batch of epochs
    # Seed is only set on the first call (when nn_prev is nothing)
    nn_current, losses, elapsed, opt_state = train_without_preprocessing(obs_uncorr, epochs_to_train;
        sigma=sigma_value,
        neurons=neurons,
        batch_size=batch_size_no_pre,
        lr=lr,
        seed=seed,
        nn_prev=nn_current,
        cumulative_time=0.0,
        opt_state_prev=opt_state)
    
    cumulative_time += elapsed
    append!(all_losses, losses)
    
    # Evaluate at this checkpoint
    score_fn = score_from_nn_scalar(nn_current, sigma_value)
    Random.seed!(111_111)
    samples_nn = generate_langevin_samples(score_fn;
        dt=dt_gen,
        n_steps=n_steps_gen,
        resolution=resolution_gen,
        n_ens=n_ens_gen,
        nn=nn_current,
        sigma=sigma_value)
    rel_ent = ScoreEstimation.relative_entropy(true_samples, samples_nn; npoints=REL_ENT_POINTS)
    push!(results_no_pre, (epochs=target_epochs, train_time=cumulative_time, relative_entropy=rel_ent, final_loss=all_losses[end]))
    
    # Save score and PDF comparison plot with loss history
    if !isempty(samples_nn)
        kde_nn = kde(samples_nn)
        fig = Plots.plot(layout=(1, 2), size=(1200, 400))
        
        # Panel 1: KDE comparison
        Plots.plot!(fig[1, 1], kde_true.x, kde_true.density;
            label="True distribution", linewidth=2, color=:steelblue,
            xlabel="x (normalized)", ylabel="density",
            title="No preprocessing: epochs=$(target_epochs)")
        Plots.plot!(fig[1, 1], kde_nn.x, kde_nn.density;
            label="NN sample", linewidth=2, linestyle=:dash, color=:darkorange)
        
        # Panel 2: Cumulative loss history
        Plots.plot!(fig[1, 2], all_losses;
            xlabel="Training epoch", ylabel="Loss",
            title="Loss function vs training epochs",
            linewidth=2, color=:steelblue, label="")
        
        pdf_path = joinpath(REDUCED_SCORE_PDFS_DIR, "no_pre_epochs_$(lpad(target_epochs, 3, '0')).png")
        Plots.savefig(fig, pdf_path)
    end
    
    GC.gc()
    @info "preprocessing=false checkpoint" epochs=target_epochs train_time=cumulative_time relative_entropy=rel_ent
end


for (i, prob) in enumerate(prob_schedule)
    # Use different seed for each independent training run
    seed = 20_000 + i
    nn, losses, t_train, nc_actual, prob_used = train_with_preprocessing(obs_uncorr, preprocessing_epochs, prob;
        sigma=sigma_value,
        neurons=neurons,
        batch_size=batch_size_pre,
        lr=lr,
        seed=seed)
    score_fn = score_from_nn_scalar(nn, sigma_value)
    Random.seed!(111_111)
    samples_nn = generate_langevin_samples(score_fn;
        dt=dt_gen,
        n_steps=n_steps_gen,
        resolution=resolution_gen,
        n_ens=n_ens_gen,
        nn=nn,
        sigma=sigma_value)
    rel_ent = ScoreEstimation.relative_entropy(true_samples, samples_nn; npoints=REL_ENT_POINTS)
    push!(results_pre, (clusters=nc_actual, clusters_actual=nc_actual, prob=prob_used, train_time=t_train, relative_entropy=rel_ent, final_loss=losses[end]))
    
    # Save score and PDF comparison plot with loss history
    if !isempty(samples_nn)
        kde_nn = kde(samples_nn)
        fig = Plots.plot(layout=(1, 2), size=(1200, 400))
        
        # Panel 1: KDE comparison
        Plots.plot!(fig[1, 1], kde_true.x, kde_true.density;
            label="True distribution", linewidth=2, color=:steelblue,
            xlabel="x (normalized)", ylabel="density",
            title="KGMM preprocessing: prob=$(prob_used), clusters=$(nc_actual)")
        Plots.plot!(fig[1, 1], kde_nn.x, kde_nn.density;
            label="NN sample", linewidth=2, linestyle=:dash, color=:darkorange)
        
        # Panel 2: Loss history for this training run
        Plots.plot!(fig[1, 2], losses;
            xlabel="Training epoch", ylabel="Loss",
            title="Loss function vs training epochs",
            linewidth=2, color=:darkorange, label="")
        
        pdf_path = joinpath(REDUCED_SCORE_PDFS_DIR, "pre_prob_$(lpad(round(Int, prob_used*1000), 3, '0')).png")
        Plots.savefig(fig, pdf_path)
    end
    
    GC.gc()
    @info "preprocessing=true" clusters=nc_actual train_time=t_train relative_entropy=rel_ent
end

# ---------------- Plotting comparison figure ----------------
train_times_no_pre = [r.train_time for r in results_no_pre]
rel_ent_no_pre = [r.relative_entropy for r in results_no_pre]
train_times_pre = [r.train_time for r in results_pre]
rel_ent_pre = [r.relative_entropy for r in results_pre]

plot_series = Plots.plot(train_times_no_pre, rel_ent_no_pre;
    marker=:circle,
    label="Without KGMM",
    xlabel="training time (s)",
    ylabel="relative entropy",
    title="Relative entropy vs training time",
    legend=:bottom)
Plots.plot!(plot_series, train_times_pre, rel_ent_pre;
    marker=:square,
    label="With KGMM")

display(plot_series)
mkpath(REDUCED_FIGURES_DIR)
Plots.savefig(plot_series, REDUCED_REL_ENT_FIG)

# ---------------- Persist results to HDF5 ----------------
mkpath(REDUCED_DATA_DIR)
h5open(REDUCED_PERF_H5, "w") do file
    write(file, "epochs_schedule", Float64.(epochs_schedule))
    write(file, "prob_schedule", Float64.(prob_schedule))
    write(file, "results_no_pre_time", Float64.(train_times_no_pre))
    write(file, "results_no_pre_relent", Float64.(rel_ent_no_pre))
    write(file, "results_pre_time", Float64.(train_times_pre))
    write(file, "results_pre_relent", Float64.(rel_ent_pre))
    write(file, "clusters_actual", Float64.([r.clusters_actual for r in results_pre]))
    write(file, "probabilities", Float64.([r.prob for r in results_pre]))
end

@info "Saved $(REDUCED_REL_ENT_FIG)"
@info "Saved $(REDUCED_PERF_H5)"

# ---------------- Human-readable summary ----------------
println("\npreprocessing = false (epochs sweep):")
for r in results_no_pre
    println("  epochs = $(r.epochs): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end

println("\npreprocessing = true (cluster sweep, epochs = $preprocessing_epochs):")
for r in results_pre
    println("  clusters = $(r.clusters) (actual $(r.clusters_actual), prob=$(round(r.prob; digits=5))): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end
