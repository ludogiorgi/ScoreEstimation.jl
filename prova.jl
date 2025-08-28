using Pkg
Pkg.activate(".")
Pkg.rm("FastSDE")                  # run once if already present
Pkg.add(url="https://github.com/ludogiorgi/FastSDE.git", rev="main")  # or a commit SHA
Pkg.status("FastSDE")
##
using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
using Revise
using ScoreEstimation
using StatsBase
using Plots
using LinearAlgebra
using Flux
using KernelDensity
using FastSDE
using StaticArrays
using BenchmarkTools

# ---------------- Problem setup ----------------

const params = (a=-0.0222, b=-0.2, c=0.0494, F_tilde=0.6, s=0.7071)

dim = 1
dt = 0.01
Nsteps = 10_000
u0 = [0.0]
resolution = 10

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
obs_uncorr = obs[:, 1:1:end]

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

# Quick visualization of the trajectory
Plots.scatter(obs_uncorr[1, 1:10000];
              markersize=2, label="",
              xlabel="X", ylabel="Y", title="Observed Trajectory")

# ---------------- KGMM call ----------------

σ_value = 0.2

@time res = ScoreEstimation.KGMM(σ_value, obs_uncorr;
                                 prob=0.02, conv_param=1e-7,
                                 show_progress=false,
                                 convention=:unit)

centers = res.centers
scores  = res.score
divs    = res.divergence
Nc      = res.Nc

# ---------------- Prepare data for plotting ----------------

centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]

# Score sorted (dim=1 → flatten)
scores_sorted = scores[:, centers_sorted_indices][:]

# Divergence is already a Vector{Float64}
divs_sorted = divs[centers_sorted_indices]

# Ground-truth values at the same (normalized) centers
scores_true = [score_true_norm(centers_sorted[i], params)[1]
               for i in eachindex(centers_sorted)]

divs_true = [divergence_true_norm(centers_sorted[i], params)[1]
             for i in eachindex(centers_sorted)]

# ---------------- Figure with two panels ----------------

plt1 = Plots.scatter(centers_sorted, scores_sorted;
                     color=:blue, markersize=3, markerstrokewidth=0,
                     label="KGMM score",
                     xlabel="x (normalized)", ylabel="score",
                     title="Score: KGMM vs Analytic")

Plots.plot!(plt1, centers_sorted, scores_true;
            color=:red, linewidth=2, label="Analytic score")

plt2 = Plots.scatter(centers_sorted, divs_sorted;
                     color=:blue, markersize=3, markerstrokewidth=0,
                     label="KGMM divergence",
                     xlabel="x (normalized)", ylabel="divergence",
                     title="Divergence: KGMM vs Analytic")

Plots.plot!(plt2, centers_sorted, divs_true;
            color=:red, linewidth=2, label="Analytic divergence")

finalfig = Plots.plot(plt1, plt2; layout=(2, 1), size=(900, 900), legend=:topright)

# Uncomment to save:
savefig(finalfig, "kgmm_score_divergence.png")
display(finalfig)
