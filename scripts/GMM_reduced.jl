cd(@__DIR__)  # Change to the script's directory
using Pkg
Pkg.activate("..")  # Activate the parent directory project
Pkg.instantiate()

using ScoreEstimation
using FastSDE
using Statistics
using KernelDensity
using Plots
using GLMakie
using HDF5
using Random

if !isdefined(ScoreEstimation, :KGMM)
    ScoreEstimation.eval(:(const KGMM = kgmm))
end

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
    du[1] = sqrt(((coeffs.A - coeffs.B * u[1])^2 + coeffs.s^2) / 2)
    return nothing
end

function score_true_physical(x::Real)
    # For SDE: dX = f(X)dt + g(X)dW
    # Stationary score: s(x) = [2f(x) - g²'(x)] / g²(x)
    # where g²(x) = [(A-Bx)² + s²]/2
    # and g²'(x) = -B(A-Bx)
    f = -coeffs.F_tilde + coeffs.a * x + coeffs.b * x^2 - coeffs.c * x^3
    g_squared = ((coeffs.A - coeffs.B * x)^2 + coeffs.s^2) / 2
    g_squared_prime = -coeffs.B * (coeffs.A - coeffs.B * x)
    numerator = 2 * f - g_squared_prime
    return numerator / g_squared
end

score_true_normalized(x::Real, mean_obs::Real, std_obs::Real) =
    std_obs * score_true_physical(x * std_obs + mean_obs)

# ---------------- Data generation ----------------
Random.seed!(1234)

dim = 1
dt = 0.01
n_steps = 10_000
resolution = 5
initial_state = zeros(dim)

obs_nn = evolve(initial_state, dt, n_steps, drift_reduced!, diffusion_reduced!;
                timestepper=:euler, resolution=resolution, sigma_inplace=true, n_ens=100)

@info "Trajectory shape" size(obs_nn)

mean_obs = mean(obs_nn)
std_obs = std(obs_nn)
obs = (obs_nn .- mean_obs) ./ std_obs
obs_uncorr = obs

# ---------------- Analytic helpers ----------------
score_true_norm_vec(x) = [score_true_normalized(xi, mean_obs, std_obs) for xi in x]
score_true_norm_scalar(x::Real) = score_true_normalized(x, mean_obs, std_obs)

# ---------------- KGMM + NN training ----------------
sigma_value = 0.05
neurons = [128, 64]
train_epochs = 2000
batch_size = 32

Plots.gr()

@time nn, train_losses, _, _, _, kgmm_res = ScoreEstimation.train(
    obs_uncorr;
    preprocessing = true,
    σ = sigma_value,
    neurons = neurons,
    n_epochs = train_epochs,
    batch_size = batch_size,
    lr = 1e-3,
    use_gpu = false,
    verbose = true,
    kgmm_kwargs = (prob=0.002, conv_param=1e-3, i_max=120, show_progress=false),
    divergence = false,
)

@info "Number of KGMM clusters" kgmm_res.Nc

function score_nn_eval(x::AbstractVector{<:Real})
    x_mat = reshape(Float32.(x), 1, :)
    vals = -Array(nn(x_mat)) ./ Float32(sigma_value)
    return Float64.(vec(vals))
end
score_nn_eval(x::Real) = score_nn_eval([x])[1]

# ---------------- Score evaluation ----------------
centers = Float64.(vec(kgmm_res.centers))
kgmm_scores = Float64.(vec(kgmm_res.score))
perm = sortperm(centers)
centers_sorted = centers[perm]
kgmm_scores_sorted = kgmm_scores[perm]
score_nn_centers = score_nn_eval(centers_sorted)
score_true_centers = score_true_norm_vec(centers_sorted)

x_min = minimum(centers_sorted)
x_max = maximum(centers_sorted)
x_grid = collect(range(x_min - 0.5, x_max + 0.5, length=400))
score_true_grid = score_true_norm_vec(x_grid)
score_nn_grid = score_nn_eval(x_grid)

plt_score = Plots.plot(x_grid, score_true_grid;
    label="Analytic score",
    color=:black,
    xlabel="x (normalized)",
    ylabel="score",
    title="Score field")
Plots.plot!(plt_score, x_grid, score_nn_grid; label="NN (interpolated)", color=:blue)
Plots.scatter!(plt_score, centers_sorted, kgmm_scores_sorted; label="KGMM centers", color=:green, ms=4)
Plots.scatter!(plt_score, centers_sorted, score_true_centers; label="Analytic @ centers", color=:red, ms=3, markerstrokewidth=0)

plt_loss = Plots.plot(1:length(train_losses), train_losses;
    yscale=:log10,
    xlabel="Epoch",
    ylabel="MSE",
    title="Training loss (log10)",
    legend=false)
##
# ---------------- Langevin generation ----------------
dt_gen = 0.005
n_steps_gen = 10_000
resolution_gen = 5

function drift_true!(du, u, t)
    du[1] = score_true_norm_scalar(u[1])
    return nothing
end

function drift_nn!(du, u, t)
    du[1] = score_nn_eval(u[1])
    return nothing
end

function unit_diffusion!(du, u, t)
    du[1] = 1.0
    return nothing
end

traj_true = evolve([0.0], dt_gen, n_steps_gen, drift_true!, unit_diffusion!;
                   timestepper=:euler, resolution=resolution_gen, sigma_inplace=true, n_ens=100)
traj_nn = evolve([0.0], dt_gen, n_steps_gen, drift_nn!, unit_diffusion!;
                 timestepper=:euler, resolution=resolution_gen, sigma_inplace=true, n_ens=100)

kde_obs = kde(vec(obs))
kde_true = kde(vec(traj_true))
kde_nn = kde(vec(traj_nn))

plt_pdf = Plots.plot(kde_obs.x, kde_obs.density;
    label="Data",
    color=:black,
    xlabel="x (normalized)",
    ylabel="PDF",
    title="Stationary distributions")
Plots.plot!(plt_pdf, kde_true.x, kde_true.density; label="Analytic Langevin", color=:red)
Plots.plot!(plt_pdf, kde_nn.x, kde_nn.density; label="NN Langevin", color=:blue)

plt_combined = Plots.plot(plt_score, plt_pdf; layout=(2, 1), size=(900, 900), legend=:topright)
mkpath("figures")
Plots.savefig(plt_combined, "figures/reduced_score_pdf.png")

plt_losses = Plots.plot(plt_loss; size=(600, 400))
Plots.savefig(plt_losses, "figures/reduced_training_loss.png")

# ---------------- Makie summary figure ----------------
mkpath("figures/GMM_figures")
mkpath("data/GMM_data")

segment_length = min(5000, size(obs, 2))
time_obs = collect(range(0, length=segment_length, step=1)) .* dt
obs_segment = Float64.(vec(obs[1, 1:segment_length]))

time_nn = collect(range(0, length=segment_length, step=1)) .* dt_gen
traj_nn_segment = Float64.(vec(traj_nn[1, 1:segment_length]))
traj_true_segment = Float64.(vec(traj_true[1, 1:segment_length]))

fig = Figure(resolution=(1200, 300), font="CMU Serif")

# Define common elements
colors = [:red, :blue]
labels = ["True", "KGMM"]

# Create subplots
ax0 = GLMakie.Axis(fig[1,1], 
    xlabel="t", ylabel="x",
    title="Observations",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax1 = GLMakie.Axis(fig[1,2], 
    xlabel="x", ylabel="Score",
    title="Scores",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax2 = GLMakie.Axis(fig[1,3], 
    xlabel="x", ylabel="PDF",
    title="PDFs",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

# Plot data
n_obs = segment_length
GLMakie.lines!(ax0, time_obs[1:10:n_obs], obs_segment[1:10:n_obs], color=:red, linewidth=1)
GLMakie.lines!(ax0, time_nn[1:10:n_obs], traj_nn_segment[1:10:n_obs], color=:blue, linewidth=1)

GLMakie.lines!(ax1, x_grid, score_true_grid, color=colors[1], linewidth=2)
GLMakie.lines!(ax1, x_grid, score_nn_grid, color=colors[2], linewidth=2)
GLMakie.xlims!(ax1, (-3, 6))
GLMakie.ylims!(ax1, (-2, 7.8))

GLMakie.lines!(ax2, kde_true.x, kde_true.density, color=colors[1], linewidth=2)
GLMakie.lines!(ax2, kde_nn.x, kde_nn.density, color=colors[2], linewidth=2)
GLMakie.xlims!(ax2, (-2.7, 7))
#GLMakie.ylims!(ax2, (0, 0.5))

# Add a more compact legend
GLMakie.Legend(fig[1, :], 
    [GLMakie.LineElement(color=c, linewidth=2) for c in colors],
    labels,
    orientation=:horizontal,
    tellheight=false,
    tellwidth=false,
    halign=:right,
    valign=:top,
    margin=(10, 10, 10, 10))

# Adjust spacing
GLMakie.colgap!(fig.layout, 20)

save("figures/GMM_figures/reduced.png", fig)

# ---------------- Persist diagnostics ----------------
centers_dataset = Float32.(centers_sorted)
score_true_dataset = Float32.(score_true_centers)
score_nn_dataset = Float32.(score_nn_centers)
kgmm_scores_dataset = Float32.(kgmm_scores_sorted)
train_losses_dataset = Float32.(train_losses)
score_true_grid_ds = Float32.(score_true_grid)
score_nn_grid_ds = Float32.(score_nn_grid)

h5open("data/GMM_data/reduced.h5", "w") do file
    write(file, "dt", dt)
    write(file, "n_steps", n_steps)
    write(file, "dt_gen", dt_gen)
    write(file, "mean_obs", mean_obs)
    write(file, "std_obs", std_obs)
    write(file, "sigma", sigma_value)
    write(file, "train_losses", train_losses_dataset)
    write(file, "centers", centers_dataset)
    write(file, "score_true_centers", score_true_dataset)
    write(file, "score_nn_centers", score_nn_dataset)
    write(file, "score_kgmm_centers", kgmm_scores_dataset)
    write(file, "x_grid", Float32.(x_grid))
    write(file, "score_true_grid", score_true_grid_ds)
    write(file, "score_nn_grid", score_nn_grid_ds)
    write(file, "kde_obs_x", Float32.(collect(kde_obs.x)))
    write(file, "kde_obs_density", Float32.(kde_obs.density))
    write(file, "kde_true_x", Float32.(collect(kde_true.x)))
    write(file, "kde_true_density", Float32.(kde_true.density))
    write(file, "kde_nn_x", Float32.(collect(kde_nn.x)))
    write(file, "kde_nn_density", Float32.(kde_nn.density))
    write(file, "obs_traj", Float32.(obs_segment))
    write(file, "traj_true", Float32.(traj_true_segment))
    write(file, "traj_nn", Float32.(traj_nn_segment))
end

@info "Saved figures/reduced_score_pdf.png"
@info "Saved figures/reduced_training_loss.png"
@info "Saved figures/GMM_figures/reduced.png"
@info "Saved data/GMM_data/reduced.h5"

# ---------------- Test relative_entropy function ----------------
@info "Testing relative_entropy function..."

# Test 1D case: compare observed data with NN-generated and analytically-generated trajectories
kl_obs_vs_nn = ScoreEstimation.relative_entropy(vec(obs), vec(traj_nn))
kl_obs_vs_true = ScoreEstimation.relative_entropy(vec(obs), vec(traj_true))
kl_true_vs_nn = ScoreEstimation.relative_entropy(vec(traj_true), vec(traj_nn))

@info "1D Relative Entropy Results:"
@info "  D_KL(obs || NN):    $kl_obs_vs_nn"
@info "  D_KL(obs || True):  $kl_obs_vs_true"
@info "  D_KL(True || NN):   $kl_true_vs_nn"

# Test multi-dimensional case: create synthetic 2D data
d = 2
n_samples = 2000
X1 = randn(d, n_samples)
X2 = randn(d, n_samples) .+ [0.5, 0.3]  # Shifted distribution

kl_vec = ScoreEstimation.relative_entropy(X1, X2)

@info "Multi-dimensional (2D) Relative Entropy Results:"
@info "  D_KL(X1 || X2) per dimension: $kl_vec"
