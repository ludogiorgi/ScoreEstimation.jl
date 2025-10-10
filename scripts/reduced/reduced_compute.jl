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
const REDUCED_DATA_DIR = joinpath(REDUCED_ROOT, "data")
const REDUCED_COMPUTE_FIG = joinpath(REDUCED_FIGURES_DIR, "reduced_compute_analysis.png")
const REDUCED_COMPUTE_H5 = joinpath(REDUCED_DATA_DIR, "reduced_compute.h5")

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
n_ens_gen = 100

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

# Convert normalized score to physical score
function score_nn_physical(x_phys::Real, nn, sigma, mean_obs, std_obs)
    x_norm = (x_phys - mean_obs) / std_obs
    score_norm = score_from_nn_scalar(nn, sigma)(x_norm)
    return score_norm / std_obs
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

# ---------------- Training with preprocessing ----------------
sigma_value = 0.05
neurons = [100, 50]
batch_size = 16
lr = 1e-3
n_epochs = 1000
prob = 0.01

Random.seed!(5000)
kgmm_kwargs = (prob=prob, conv_param=1e-4, i_max=120, show_progress=true)

@info "Training neural network score with preprocessing=true"
nn, losses, _, _, _, res = ScoreEstimation.train(
    obs_uncorr;
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

@info "Training completed" clusters=res.Nc final_loss=losses[end]

# ---------------- Generate samples from NN score ----------------
score_fn = score_from_nn_scalar(nn, sigma_value)
Random.seed!(111_111)

@info "Generating samples from NN score"
samples_nn = generate_langevin_samples(score_fn;
    dt=dt_gen,
    n_steps=n_steps_gen,
    resolution=resolution_gen,
    n_ens=n_ens_gen,
    nn=nn,
    sigma=sigma_value)

kde_nn = kde(samples_nn)

# ---------------- Unnormalize data for plotting/saving ----------------
# Unnormalize samples
true_samples_phys = true_samples .* std_obs .+ mean_obs
samples_nn_phys = samples_nn .* std_obs .+ mean_obs

# Compute PDFs in physical space
kde_true_phys = kde(true_samples_phys)
kde_nn_phys = kde(samples_nn_phys)
kde_emp_phys = kde(vec(obs_nn))

# ---------------- Compute comparison data ----------------
# Grid for score and PDF comparison (physical space)
x_min_phys = min(minimum(obs_nn), minimum(true_samples_phys), minimum(samples_nn_phys))
x_max_phys = max(maximum(obs_nn), maximum(true_samples_phys), maximum(samples_nn_phys))
x_grid_phys = range(x_min_phys, x_max_phys; length=400)

# Compute score functions in physical space
score_true_vals_phys = score_true_physical.(x_grid_phys)
score_nn_vals_phys = [score_nn_physical(x, nn, sigma_value, mean_obs, std_obs) for x in x_grid_phys]

# Relative entropy
rel_ent = ScoreEstimation.relative_entropy(true_samples, samples_nn; npoints=REL_ENT_POINTS)
@info "Relative entropy" rel_ent=rel_ent

# ---------------- Plotting with 3 subplots (physical space) ----------------
Plots.gr()
mkpath(REDUCED_FIGURES_DIR)

fig = Plots.plot(layout=(3, 1), size=(900, 1200))

# Subplot 1: Trajectory comparison (obs vs generated) - physical space
plot_len = min(5000, length(obs_nn), length(samples_nn_phys))
Plots.plot!(fig, 1:plot_len, obs_nn[1:plot_len];
    label="observed trajectory",
    xlabel="time step",
    ylabel="x",
    title="Trajectory Comparison",
    linewidth=1.5,
    alpha=0.7,
    subplot=1,
    legend=:topright)
Plots.plot!(fig, 1:plot_len, samples_nn_phys[1:plot_len];
    label="NN score trajectory",
    linewidth=1.5,
    alpha=0.7,
    linestyle=:dash,
    subplot=1)

# Subplot 2: Score function comparison - physical space
Plots.plot!(fig, x_grid_phys, score_true_vals_phys;
    label="analytic score",
    xlabel="x",
    ylabel="score s(x)",
    title="Score Function Comparison",
    linewidth=2,
    subplot=2,
    legend=:topright)
Plots.plot!(fig, x_grid_phys, score_nn_vals_phys;
    label="NN score",
    linewidth=2,
    linestyle=:dash,
    subplot=2)

# Subplot 3: PDF comparison - physical space
Plots.plot!(fig, kde_true_phys.x, kde_true_phys.density;
    label="reference PDF (analytic score)",
    xlabel="x",
    ylabel="density",
    title="PDF Comparison (preprocessing=true, clusters=$(res.Nc), rel. ent.=$(round(rel_ent, digits=4)))",
    linewidth=2,
    subplot=3,
    legend=:topright)
Plots.plot!(fig, kde_nn_phys.x, kde_nn_phys.density;
    label="NN score PDF",
    linewidth=2,
    linestyle=:dash,
    subplot=3)
Plots.plot!(fig, kde_emp_phys.x, kde_emp_phys.density;
    label="empirical PDF (obs)",
    linewidth=2,
    linestyle=:dot,
    color=:gray,
    subplot=3)

display(fig)
Plots.savefig(fig, REDUCED_COMPUTE_FIG)

# ---------------- Save all data to HDF5 ----------------
mkpath(REDUCED_DATA_DIR)

h5open(REDUCED_COMPUTE_H5, "w") do file
    # Training data (normalized for reference)
    write(file, "obs_normalized", Float32.(obs))
    write(file, "mean_obs", Float64(mean_obs))
    write(file, "std_obs", Float64(std_obs))

    # Physical space data (what we plot/analyze)
    write(file, "obs_nn", Float32.(obs_nn))
    write(file, "true_samples", Float32.(true_samples_phys))
    write(file, "samples_nn", Float32.(samples_nn_phys))

    # Trajectory comparison (first plot_len points) - physical space
    write(file, "trajectory_obs", Float32.(obs_nn[1:plot_len]))
    write(file, "trajectory_nn", Float32.(samples_nn_phys[1:plot_len]))

    # Score function comparison - physical space
    write(file, "x_grid", Float32.(x_grid_phys))
    write(file, "score_true", Float32.(score_true_vals_phys))
    write(file, "score_nn", Float32.(score_nn_vals_phys))

    # PDF comparison - physical space
    write(file, "pdf_true_x", Float32.(kde_true_phys.x))
    write(file, "pdf_true_density", Float32.(kde_true_phys.density))
    write(file, "pdf_nn_x", Float32.(kde_nn_phys.x))
    write(file, "pdf_nn_density", Float32.(kde_nn_phys.density))
    write(file, "pdf_emp_x", Float32.(kde_emp_phys.x))
    write(file, "pdf_emp_density", Float32.(kde_emp_phys.density))

    # Metadata
    write(file, "n_clusters", Int(res.Nc))
    write(file, "relative_entropy", Float64(rel_ent))
    write(file, "final_loss", Float64(losses[end]))
    write(file, "sigma", Float64(sigma_value))
    write(file, "n_epochs", Int(n_epochs))
end

@info "Saved $(REDUCED_COMPUTE_FIG)"
@info "Saved $(REDUCED_COMPUTE_H5)"

println("\nAnalysis summary:")
println("  Clusters: $(res.Nc)")
println("  Relative entropy: $(round(rel_ent; digits=4))")
println("  Final loss: $(round(losses[end]; digits=6))")
