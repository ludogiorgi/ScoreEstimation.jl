using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ScoreEstimation
using Plots
using Statistics
using KernelDensity
using FastSDE

# ---------------- 3D Triad SDE ----------------
const params = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

dim = 3
dt = 0.01
Nsteps = 10_000
u0 = [0.0, 0.0, 0.0]
resolution = 10

function drift!(du, u, t)
    du[1] = -params.dᵤ * u[1] - params.wᵤ * u[2] + u[3]
    du[2] = -params.dᵤ * u[2] + params.wᵤ * u[1]
    du[3] = -params.dₜ * u[3]
end

function diffusion!(du, u, t)
    du[1] = params.σ₁
    du[2] = params.σ₂
    du[3] = 1.5 * (tanh(u[1]) + 1)
end

obs_nn = evolve(u0, dt, Nsteps, drift!, diffusion!;
                timestepper=:euler, resolution=resolution, n_ens=100)

@info "Trajectory shape: $(size(obs_nn))"

# Normalize observations
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S
obs_uncorr = obs

# ---------------- Training ----------------
σ_value = 0.05
neurons = [dim, 128, 64, dim]

# Preprocessing=true (KGMM + weighted interpolation)
nn_k, losses_k, _, div_k, kgmm = ScoreEstimation.train(
    obs_uncorr; preprocessing=true, σ=σ_value, neurons=neurons,
    n_epochs=1000, batch_size=32, lr=1e-3, use_gpu=false, verbose=true,
    kgmm_kwargs=(prob=0.001, conv_param=1e-2, i_max=100, show_progress=false),
    divergence=true, probes=1)
sθ_k = X -> -nn_k(Float32.(X)) ./ Float32(σ_value)

# Preprocessing=false (raw DSM)
nn_r, losses_r, _, div_r, _ = ScoreEstimation.train(
    obs_uncorr; preprocessing=false, σ=σ_value, neurons=neurons,
    n_epochs=100, batch_size=32, lr=1e-3, use_gpu=false, verbose=true,
    divergence=true, probes=1)
sθ_r = X -> -nn_r(Float32.(X)) ./ Float32(σ_value)

# ---------------- Score-based generation (normalized space) ----------------
σ_langevin = fill(sqrt(2.0), dim)
dt_gen = 0.005
Nsteps_gen = 10_000
resolution_gen = 10

drift_score_k!(du, u, t) = (du .= Float64.(sθ_k(u)))
drift_score_r!(du, u, t) = (du .= Float64.(sθ_r(u)))

traj_k = evolve(zeros(dim), dt_gen, Nsteps_gen, drift_score_k!, σ_langevin;
                timestepper=:euler, resolution=resolution_gen, n_ens=100)
traj_r = evolve(zeros(dim), dt_gen, Nsteps_gen, drift_score_r!, σ_langevin;
                timestepper=:euler, resolution=resolution_gen, n_ens=100)

# ---------------- PDFs: true vs NN+KGMM vs NN (DSM) ----------------
gr()
plots_pdf = Vector{Any}(undef, 3)
labels = ("X", "Y", "Z")
for j in 1:3
    kde_true = kde(obs_uncorr[j, :])
    kde_k    = kde(traj_k[j, :])
    kde_r    = kde(traj_r[j, :])
    plt = plot(kde_true.x, kde_true.density; label="True", lw=2,
               xlabel=labels[j], ylabel="Density", title="$(labels[j]) marginal PDF")
    plot!(plt, kde_k.x, kde_k.density; label="NN+KGMM", lw=2)
    plot!(plt, kde_r.x, kde_r.density; label="NN (DSM)", lw=2)
    plots_pdf[j] = plt
end
fig_pdf = plot(plots_pdf...; layout=(3,1), size=(900,900), legend=:topright)
savefig(fig_pdf, "figures/triad_pdfs.png")

# ---------------- Divergence comparison vs variables ----------------
# Sample evaluation points from the true (normalized) data
Neval = min(5000, size(obs_uncorr,2))
idx = rand(1:size(obs_uncorr,2), Neval)
Xeval = obs_uncorr[:, idx]
div_k_vals = vec(div_k(Xeval))
div_r_vals = vec(div_r(Xeval))

plots_div = Vector{Any}(undef, 3)
for j in 1:3
    xj = vec(Xeval[j, :])
    plt = scatter(xj, div_k_vals; ms=2, alpha=0.4, label="NN+KGMM",
                  xlabel=labels[j], ylabel="divergence",
                  title="Divergence vs $(labels[j])")
    scatter!(plt, xj, div_r_vals; ms=2, alpha=0.4, label="NN (DSM)")
    plots_div[j] = plt
end
fig_div = plot(plots_div...; layout=(3,1), size=(900,900), legend=:topright)
savefig(fig_div, "figures/triad_divergences.png")

@info "Saved: figures/triad_pdfs.png and figures/triad_divergences.png"
