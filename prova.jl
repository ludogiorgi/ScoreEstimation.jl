# using Pkg
# Pkg.activate(".")
# Pkg.rm("FastSDE")                  # run once if already present
# Pkg.add(url="https://github.com/ludogiorgi/FastSDE.git", rev="main")  # or a commit SHA
# Pkg.status("FastSDE")
# ##
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ScoreEstimation
using Plots
using LinearAlgebra
using Statistics
using Flux
using FastSDE
using Statistics
##
# ---------------- Problem setup ----------------

const params = (a=-0.0222, b=-0.2, c=0.0494, F_tilde=0.6, s=0.7071)

dim = 1
dt = 0.01
Nsteps = 10_000
u0 = [0.0]
resolution = 5

function drift!(du, u, t)
    du[1] = params.F_tilde + params.a*u[1] + params.b*u[1]^2 - params.c*u[1]^3
end

function sigma!(du, u, t)
    du[1] = params.s
end

obs_nn = evolve(u0, dt, Nsteps, drift!, sigma!;
                timestepper=:euler, resolution=resolution,
                sigma_inplace=true, n_ens=100)

println("FastSDE trajectory shape: ", size(obs_nn))

# ---------------- Normalization ----------------

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S
obs_uncorr = obs[:, :]

# ---------------- Ground-truth (steady-state convention) ----------------

# Score: s(x) = (F_tilde + a x + b x^2 - c x^3) / s^2
function score_true(x, p=params)
    u = x[1]
    return [(p.F_tilde + p.a*u + p.b*u^2 - p.c*u^3) / (p.s^2)]
end

# Divergence in 1D is d/dx of the score:
# div s(x) = (a + 2 b x - 3 c x^2) / s^2
function divergence_true(x, p=params)
    u = x[1]
    return [(p.a + 2*p.b*u - 3*p.c*u^2) / (p.s^2)]
end

# Normalization helpers (map normalized coordinate ξ to physical x = ξ*S + M)
# Score transforms as s_norm(ξ) = S * s(x)
function normalize_f(f, x, M, S, p=params)
    return (f(x .* S .+ M, p) .* S)[:]
end

# Divergence transforms in 1D as div_norm(ξ) = S^2 * div(x)
function normalize_divergence(fdiv, x, M, S, p=params)
    return (fdiv(x .* S .+ M, p) .* (S .^ 2))[:]
end

score_true_norm(x, p=params)       = normalize_f(score_true, x, M, S, p)
divergence_true_norm(x, p=params)  = normalize_divergence(divergence_true, x, M, S, p)

# Diffusion noise (scalar)
σ_value = 0.05

############## Train NN (DSM) and get divergence via Hutchinson ##############

neurons = [size(obs_uncorr,1), 100, 50, size(obs_uncorr,1)]  # D -> 100 -> 50 -> D

# Train ε-net on-the-fly draws; ask wrapper for divergence and return KGMM when preprocessing=true
@time nn, train_losses, _, div_fn, res_wrapped = ScoreEstimation.train(
    obs_uncorr;
    preprocessing = true,           # use KGMM preprocessing
    σ             = σ_value,
    kgmm_kwargs   = (prob=0.002, conv_param=1e-4, show_progress=false, convention=:unit),
    n_epochs      = 2000,
    batch_size    = 16,
    lr            = 1e-3,
    neurons       = neurons,
    divergence    = true,
    probes        = 1,
    use_gpu       = false,
    verbose       = false,
)

# score from ε-net: for :unit convention, sθ(x) = -ε̂(x)/(2σ)
sθ = X -> -Array(nn(Float32.(X))) ./ (2f0 * Float32(σ_value))

# Use KGMM output returned by the training wrapper
centers = res_wrapped.centers
scores  = res_wrapped.score
divs    = res_wrapped.divergence

# Prepare data for plotting (sorted by x)
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores_sorted = scores[:, centers_sorted_indices][:]
divs_sorted = divs[centers_sorted_indices]

# Ground-truth values at the same (normalized) centers
scores_true = [score_true_norm(centers_sorted[i], params)[1] for i in eachindex(centers_sorted)]
divs_true   = [divergence_true_norm(centers_sorted[i], params)[1] for i in eachindex(centers_sorted)]

# Predict at KGMM centers
Yscore_hat = sθ(centers)                     # (D, C)
Ydiv_hat   = div_fn(centers)                 # (1, C)

score_nn_sorted  = vec(Yscore_hat)[centers_sorted_indices]
div_nn_sorted    = vec(Ydiv_hat)[centers_sorted_indices]

# Plots
plt1 = Plots.plot(centers_sorted, score_nn_sorted;
                     color=:blue, lw=2, label="NN (score)",
                     xlabel="x (normalized)", ylabel="score",
                     title="Score: NN vs Analytic & KGMM")
Plots.plot!(plt1, centers_sorted, scores_true; color=:red, lw=2, label="Analytic score")
Plots.scatter!(plt1, centers_sorted, scores_sorted; color=:green, ms=3, msw=0, label="KGMM score (centers)")

plt2 = Plots.plot(centers_sorted, div_nn_sorted;
                     color=:blue, lw=2, label="NN (divergence)",
                     xlabel="x (normalized)", ylabel="divergence",
                     title="Divergence: NN vs Analytic & KGMM")
Plots.plot!(plt2, centers_sorted, divs_true; color=:red, lw=2, label="Analytic divergence")
Plots.scatter!(plt2, centers_sorted, divs_sorted; color=:green, ms=3, msw=0, label="KGMM divergence (centers)")

finalfig = Plots.plot(plt1, plt2; layout=(2, 1), size=(900, 900), legend=:topright)
display(finalfig)
savefig(finalfig, "figures/nn_vs_truth.png")

# ---------------- RMSE diagnostics ----------------
rmse(a, b) = sqrt(mean((Float64.(a) .- Float64.(b)).^2))

score_true_v  = Float64.(scores_true)
score_nn_v    = Float64.(score_nn_sorted)
score_kgmm_v  = Float64.(vec(scores_sorted))

div_true_v    = Float64.(divs_true)
div_nn_v      = Float64.(div_nn_sorted)
div_kgmm_v    = Float64.(divs_sorted)

rmse_s_nn_true   = rmse(score_nn_v,   score_true_v)
rmse_s_kg_true   = rmse(score_kgmm_v, score_true_v)
rmse_s_nn_kg     = rmse(score_nn_v,   score_kgmm_v)

rmse_d_nn_true   = rmse(div_nn_v,   div_true_v)
rmse_d_kg_true   = rmse(div_kgmm_v, div_true_v)
rmse_d_nn_kg     = rmse(div_nn_v,   div_kgmm_v)

@info "RMSE (score):  NN vs True = $(rmse_s_nn_true),  KGMM vs True = $(rmse_s_kg_true),  NN vs KGMM = $(rmse_s_nn_kg)"
@info "RMSE (divergence):  NN vs True = $(rmse_d_nn_true),  KGMM vs True = $(rmse_d_kg_true),  NN vs KGMM = $(rmse_d_nn_kg)"
