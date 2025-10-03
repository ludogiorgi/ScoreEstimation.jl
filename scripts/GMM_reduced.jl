using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ScoreEstimation
using FastSDE
using Statistics
using Plots
using Flux
using HDF5
using Random
using KernelDensity

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
n_steps = 100_000
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
    generate_langevin_samples(score_scalar_fn; dt, n_steps, resolution, n_ens, boundary=(-5,5))

Sample using FastSDE.evolve with drift = score and diffusion = sqrt(2) (normalized space).
"""
function generate_langevin_samples(score_scalar_fn; dt, n_steps, resolution, n_ens, boundary=(-10,10))
    function drift!(du, u, t)
        du[1] = score_scalar_fn(u[1])
        return nothing
    end
    function unit_diffusion!(du, u, t)
        du[1] = sqrt(2.0)
        return nothing
    end
    samples = evolve([0.0], dt, n_steps, drift!, unit_diffusion!;
                     timestepper=:euler, resolution=resolution,
                     sigma_inplace=true, n_ens=n_ens, boundary=boundary)
    return vec(samples)
end

Random.seed!(42)
dt_gen = 0.005
n_steps_gen = 2_000_000
resolution_gen = 200
n_ens_gen = 1

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

function train_without_preprocessing(obs, epochs; sigma, neurons, batch_size, lr, seed)
    Random.seed!(seed)
    nn = nothing
    losses = nothing
    elapsed = @elapsed begin
        nn, losses, _, _, _, _ = ScoreEstimation.train(
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
        )
    end
    return nn, losses, elapsed
end

function prob_for_clusters(nc::Int)
    if nc <= 60
        return 0.012
    elseif nc <= 120
        return 0.006
    elseif nc <= 180
        return 0.004
    elseif nc <= 240
        return 0.0029
    else
        return 0.0025
    end
end

function train_with_preprocessing(obs, epochs, nc_target;
        sigma, neurons, batch_size, lr, seed,
        conv_param=1e-4, i_max=120, show_progress=false)
    Random.seed!(seed)
    prob = prob_for_clusters(nc_target)
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

epochs_schedule = [5, 10, 15, 20]
clusters_schedule = [60, 120, 180, 240, 300]
preprocessing_epochs = 2000

Plots.gr()

mkpath("figures/score_pdfs")

# Warm-up runs to remove compilation bias from first measurements
@info "Warm-up (preprocessing=false)"
train_without_preprocessing(obs_uncorr, 1;
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=9999)

@info "Warm-up (preprocessing=true)"
train_with_preprocessing(obs_uncorr, 1, clusters_schedule[1];
    sigma=sigma_value,
    neurons=neurons,
    batch_size=batch_size,
    lr=lr,
    seed=19_999)

# ---------------- Sweeps ----------------
results_no_pre = Vector{NamedTuple}(undef, length(epochs_schedule))
results_pre = Vector{NamedTuple}(undef, length(clusters_schedule))

for (i, epochs) in enumerate(epochs_schedule)
    seed = 10_000 + epochs
    nn, losses, t_train = train_without_preprocessing(obs_uncorr, epochs;
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
        n_ens=n_ens_gen)
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
    Plots.savefig(iter_fig, "figures/score_pdfs/pre_false_epochs_$(epochs).png")
    results_no_pre[i] = (epochs=epochs, train_time=t_train, relative_entropy=rel_ent, final_loss=losses[end])
    GC.gc()
    @info "preprocessing=false" epochs=epochs train_time=t_train relative_entropy=rel_ent
end

for (i, nc) in enumerate(clusters_schedule)
    seed = 20_000 + nc
        nn, losses, t_train, nc_actual, prob_used = train_with_preprocessing(obs_uncorr, preprocessing_epochs, nc;
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
        n_ens=n_ens_gen)
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
        title="Score PDFs (preprocessing = true, clusters = $(nc))",
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
    Plots.savefig(iter_fig, "figures/score_pdfs/pre_true_clusters_$(nc).png")
    results_pre[i] = (clusters=nc, clusters_actual=nc_actual, prob=prob_used, train_time=t_train, relative_entropy=rel_ent, final_loss=losses[end])
    GC.gc()
    @info "preprocessing=true" clusters=nc train_time=t_train relative_entropy=rel_ent
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
mkpath("figures")
Plots.savefig(plot_series, "figures/reduced_relative_entropy_vs_time.png")

# ---------------- Persist results ----------------
mkpath("data/GMM_data")

h5open("data/GMM_data/reduced_comparison.h5", "w") do file
    write(file, "epochs_schedule", Float64.(epochs_schedule))
    write(file, "clusters_schedule", Float64.(clusters_schedule))
    write(file, "results_no_pre_time", Float64.(train_times_no_pre))
    write(file, "results_no_pre_relent", Float64.(rel_ent_no_pre))
    write(file, "results_pre_time", Float64.(train_times_pre))
    write(file, "results_pre_relent", Float64.(rel_ent_pre))
    write(file, "clusters_actual", Float64.([r.clusters_actual for r in results_pre]))
    write(file, "probabilities", Float64.([r.prob for r in results_pre]))
    write(file, "true_samples", Float32.(true_samples))
end

@info "Saved figures/reduced_relative_entropy_vs_time.png"
@info "Saved data/GMM_data/reduced_comparison.h5"

# ---------------- Human-readable summary ----------------
println("\npreprocessing = false (epochs sweep):")
for r in results_no_pre
    println("  epochs = $(r.epochs): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end

println("\npreprocessing = true (cluster sweep, epochs = $preprocessing_epochs):")
for r in results_pre
    println("  clusters = $(r.clusters) (actual $(r.clusters_actual), prob=$(round(r.prob; digits=5))): time = $(round(r.train_time; digits=3)) s, KL = $(round(r.relative_entropy; digits=4)), final loss = $(round(r.final_loss; digits=6))")
end
