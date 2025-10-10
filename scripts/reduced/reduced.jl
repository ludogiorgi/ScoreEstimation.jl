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

const REDUCED_ROOT = @__DIR__
const REDUCED_FIGURES_DIR = joinpath(REDUCED_ROOT, "figures")
const REDUCED_SCORE_PDFS_DIR = joinpath(REDUCED_FIGURES_DIR, "score_pdfs")
const REDUCED_DATA_DIR = joinpath(REDUCED_ROOT, "data", "GMM_data")

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
                n_burnin=1000, timestepper=:euler, resolution=resolution, sigma_inplace=true, n_ens=100, boundary=(-10,10))

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
function generate_langevin_samples(score_scalar_fn; dt, n_steps, resolution, n_ens, boundary=(-10,10), nn=nothing, sigma=nothing)
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
                          n_burnin=1000,
                          n_ens=n_ens,
                          boundary=boundary,
                          batched_drift=true,
                          manage_blas_threads=true)
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
                         n_burnin=1000,
                         sigma_inplace=true,
                         n_ens=n_ens,
                         boundary=boundary,
                         batched_drift=false,
                         manage_blas_threads=false)
    end
    return vec(samples)
end

Random.seed!(42)
dt_gen = 0.005
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
function create_batched_drift_nn(nn, sigma)
    # Primary 4-arg method (DU,U,p,t)
    function drift_batched!(DU, U, p, t)
        U_f32 = Float32.(U)          # U: (1, n_ens)
        Y = nn(U_f32)                # same shape
        DU .= -Float64.(Y) ./ Float64(sigma)
        return nothing
    end
    # 3-arg fallback used by static path probes (DU,U,t)
    function drift_batched!(DU, U, t)
        drift_batched!(DU, U, nothing, t)
        return nothing
    end
    return drift_batched!
end

function train_without_preprocessing(obs, epochs; sigma, neurons, batch_size, lr, seed,
                                     nn_prev=nothing, cumulative_time=0.0, opt_state_prev=nothing)
    Random.seed!(seed)
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
sigma_value = 0.05
neurons = [100, 50]
batch_size = 16
lr = 1e-3

epochs_schedule = [10:10:60...]
prob_schedule = [0.01:-0.002:0.002...]
preprocessing_epochs = 1000

Plots.gr()

mkpath(REDUCED_SCORE_PDFS_DIR)

# Warm-up runs to remove compilation bias from first measurements
@info "Warm-up (preprocessing=false)"
train_without_preprocessing(obs_uncorr, 1;
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=9999)

@info "Warm-up (preprocessing=true)"
train_with_preprocessing(obs_uncorr, 1, prob_schedule[1];
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=19_999)

# ---------------- Sweeps ----------------
# Result containers (dynamic). We use concrete NamedTuple types so downstream
# comprehensions produce concretely typed arrays (avoids Vector{Any} and HDF5 write errors).
const NoPreResult = NamedTuple{(:epochs,:train_time,:relative_entropy,:final_loss),Tuple{Int,Float64,Float64,Float64}}
const PreResult   = NamedTuple{(:clusters,:clusters_actual,:prob,:train_time,:relative_entropy,:final_loss),Tuple{Int,Int,Float64,Float64,Float64,Float64}}
results_no_pre = NoPreResult[]
results_pre    = PreResult[]


prev_nn = nothing
prev_opt_state = nothing
cumulative_time_no_pre = 0.0
for (i, epochs) in enumerate(epochs_schedule)
    global prev_nn, prev_opt_state, cumulative_time_no_pre
    seed = 10_000 + i # fixed seed for consistent incremental training
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
    kde_nn = kde(samples_nn)
    x_min = min(minimum(true_samples), minimum(samples_nn))
    x_max = max(maximum(true_samples), maximum(samples_nn))
    x_grid = range(x_min, x_max; length=400)
    score_true_vals = score_true_norm_scalar.(x_grid)
    score_nn_vals = map(score_fn, x_grid)
    iter_fig = Plots.plot(layout=(2, 1), size=(900, 700))
    # empirical KDE from normalized observations
    kde_emp = kde(vec(obs))
    Plots.plot!(iter_fig, kde_true.x, kde_true.density;
        label="reference pdf (analytic score)",
        xlabel="x",
        ylabel="density",
        title="Score PDFs (preprocessing = false, epochs = $(epochs))",
        linewidth=2,
        subplot=1,
        legend=:topright)
    Plots.plot!(iter_fig, kde_nn.x, kde_nn.density;
        label="pdf (nn score)",
        linewidth=2,
        linestyle=:dash,
        subplot=1)
    Plots.plot!(iter_fig, kde_emp.x, kde_emp.density;
        label="empirical pdf (obs)",
        linewidth=2,
        linestyle=:dot,
        color=:gray,
        subplot=1)
    Plots.plot!(iter_fig, x_grid, score_true_vals;
        label="analytic score",
        xlabel="x",
        ylabel="score",
        title="Score Functions",
        linewidth=2,
        subplot=2,
        legend=:topright)
    Plots.plot!(iter_fig, x_grid, score_nn_vals;
        label="nn score",
        linewidth=2,
        linestyle=:dash,
        subplot=2)
    display(iter_fig)
    Plots.savefig(iter_fig, joinpath(REDUCED_SCORE_PDFS_DIR, "pre_false_epochs_$(epochs).png"))
    push!(results_no_pre, (epochs=epochs, train_time=t_train, relative_entropy=rel_ent, final_loss=losses[end]))
    GC.gc()
    @info "preprocessing=false" epochs=epochs train_time=t_train relative_entropy=rel_ent
end

for (i, prob) in enumerate(prob_schedule)
    seed = 20_000 + i
    nn, losses, t_train, nc_actual, prob_used = train_with_preprocessing(obs_uncorr, preprocessing_epochs, prob;
        sigma=sigma_value,
        neurons=neurons,
        batch_size=batch_size,
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
    kde_nn = kde(samples_nn)
    x_min = min(minimum(true_samples), minimum(samples_nn))
    x_max = max(maximum(true_samples), maximum(samples_nn))
    x_grid = range(x_min, x_max; length=400)
    score_true_vals = score_true_norm_scalar.(x_grid)
    score_nn_vals = map(score_fn, x_grid)
    iter_fig = Plots.plot(layout=(2, 1), size=(900, 700))
    kde_emp = kde(vec(obs))
    Plots.plot!(iter_fig, kde_true.x, kde_true.density;
        label="reference pdf (analytic score)",
        xlabel="x",
        ylabel="density",
        title="Score PDFs (preprocessing = true, # clusters = $(nc_actual))",
        linewidth=2,
        subplot=1,
        legend=:topright)
    Plots.plot!(iter_fig, kde_nn.x, kde_nn.density;
        label="pdf (nn score)",
        linewidth=2,
        linestyle=:dash,
        subplot=1)
    Plots.plot!(iter_fig, kde_emp.x, kde_emp.density;
        label="empirical pdf (obs)",
        linewidth=2,
        linestyle=:dot,
        color=:gray,
        subplot=1)
    Plots.plot!(iter_fig, x_grid, score_true_vals;
        label="analytic score",
        xlabel="x",
        ylabel="score",
        title="Score Functions",
        linewidth=2,
        subplot=2,
        legend=:topright)
    Plots.plot!(iter_fig, x_grid, score_nn_vals;
        label="nn score",
        linewidth=2,
        linestyle=:dash,
        subplot=2)
    display(iter_fig)
    Plots.savefig(iter_fig, joinpath(REDUCED_SCORE_PDFS_DIR, "pre_true_clusters_$(nc_actual).png"))
    push!(results_pre, (clusters=nc_actual, clusters_actual=nc_actual, prob=prob_used, train_time=t_train, relative_entropy=rel_ent, final_loss=losses[end]))
    GC.gc()
    @info "preprocessing=true" clusters=nc_actual train_time=t_train relative_entropy=rel_ent
end

# ---------------- Plotting ----------------
train_times_no_pre = [r.train_time for r in results_no_pre]
rel_ent_no_pre = [r.relative_entropy for r in results_no_pre]
train_times_pre = [r.train_time for r in results_pre]
rel_ent_pre = [r.relative_entropy for r in results_pre]

plot_series = Plots.plot(train_times_no_pre, rel_ent_no_pre;
    marker=:circle,
    label="preprocessing = false",
    xlabel="training time (s)",
    ylabel="relative entropy",
    title="Relative entropy vs training time",
    legend=:topright)
Plots.plot!(plot_series, train_times_pre, rel_ent_pre;
    marker=:square,
    label="preprocessing = true")

display(plot_series)
mkpath(REDUCED_FIGURES_DIR)
relative_entropy_fig_path = joinpath(REDUCED_FIGURES_DIR, "reduced_relative_entropy_vs_time.png")
Plots.savefig(plot_series, relative_entropy_fig_path)

# ---------------- Persist results ----------------
mkpath(REDUCED_DATA_DIR)

h5open(joinpath(REDUCED_DATA_DIR, "reduced_comparison.h5"), "w") do file
    write(file, "epochs_schedule", Float64.(epochs_schedule))
    write(file, "prob_schedule", Float64.(prob_schedule))
    write(file, "results_no_pre_time", Float64.(train_times_no_pre))
    write(file, "results_no_pre_relent", Float64.(rel_ent_no_pre))
    write(file, "results_pre_time", Float64.(train_times_pre))
    write(file, "results_pre_relent", Float64.(rel_ent_pre))
    write(file, "clusters_actual", Float64.([r.clusters_actual for r in results_pre]))
    write(file, "probabilities", Float64.([r.prob for r in results_pre]))
    write(file, "true_samples", Float32.(true_samples))
end

@info "Saved $(relative_entropy_fig_path)"
@info "Saved $(joinpath(REDUCED_DATA_DIR, "reduced_comparison.h5"))"

# ---------------- Human-readable summary ----------------
println("\npreprocessing = false (epochs sweep):")
for r in results_no_pre
    println("  epochs = $(r.epochs): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end

println("\npreprocessing = true (cluster sweep, epochs = $preprocessing_epochs):")
for r in results_pre
    println("  clusters = $(r.clusters) (actual $(r.clusters_actual), prob=$(round(r.prob; digits=5))): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end