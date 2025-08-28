using Pkg
Pkg.activate(".")
Pkg.rm("FastSDE")                  # run once if already present
Pkg.add(url="https://github.com/ludogiorgi/FastSDE.git", rev="main")  # or a commit SHA
Pkg.status("FastSDE")
##
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ScoreEstimation
using StatsBase
using Plots
using LinearAlgebra
using Flux
using KernelDensity
using DifferentialEquations
using FastSDE
using StaticArrays
using BenchmarkTools

##

# Define parameters as a named tuple
const params = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

dim = 3
dt = 0.01
Nsteps = 1000000
u0 = [0.0, 0.0, 0.0]

# Solve the SDE using FastSDE evolve API, saving every 'resolution' steps
resolution = 10


# Define the SDE problem for DifferentialEquations
function drift!(du, u, p, t)
    du[1] = -params.dᵤ * u[1] - params.wᵤ * u[2] + u[3]
    du[2] = -params.dᵤ * u[2] + params.wᵤ * u[1]
    du[3] = -params.dₜ * u[3]
end

function diffusion!(du, u, p, t)
    du[1] = params.σ₁
    du[2] = params.σ₂
    du[3] = 1.5 * (tanh(u[1]) + 1)
end

# Create the SDE problem

traj = evolve(u0, dt, 100*Nsteps, drift!, diffusion!; params=params, timestepper=:euler, resolution=resolution, n_ens=1)
##
# Convert solution to matrix format for comparison


println("FastSDE trajectory shape: ", size(traj))
println("DifferentialEquations trajectory shape: ", size(traj_de))

# Extract the trajectory as a matrix (dim x time steps)
obs_nn = traj

# Normalize the observations
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

plot(obs[1,1:1000])

# Compute autocovariance using StatsBase
autocov_obs = zeros(dim, 300)
lags = 0:299  # Lags from 0 to 299 for 300 timesteps
for i in 1:dim
    autocov_obs[i, :] = autocov(obs[i, :], lags; demean=true)
end

# Mean autocovariance across dimensions
autocov_obs_mean = mean(autocov_obs, dims=1)

# Plot the mean autocovariance
Plots.plot(autocov_obs_mean[1, :], label="Mean", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
############################ SCORE ESTIMATION (USING MAIN MODULE) ####################

σ_value = 0.05

# Choose method: :kgmm or :vanilla
method = :kgmm

# Estimate score using the selected method (functions provided by the main module)
if method == :kgmm
    # Use raw (unnormalized) data; normalization is handled internally
    result = calculate_score_kgmm(
        obs_nn;
        σ_value=σ_value,
        clustering_prob=0.0001,
        clustering_conv_param=0.002,
        clustering_max_iter=150,
        use_normalization_for_clustering=false,
        epochs=200,
        batch_size=16,
        hidden_layers=[100, 50],
        use_gpu=false,
        verbose=true,
    )
elseif method == :vanilla
    result = calculate_score_vanilla(
        obs_nn;
        σ_value=σ_value,
        epochs=200,
        batch_size=16,
        hidden_layers=[100, 50],
        use_gpu=false,
        verbose=true,
    )
else
    error("Unknown method $(method). Use :kgmm or :vanilla.")
end

score_fn = result.score_function

##
#################### SAMPLES GENERATION ####################

# Wrap score function into an in-place drift for FastSDE
function drift_score!(du, u, t)
    du .= Float64.(score_fn(u))
end


# Define constant diffusion for score-based generation
σ_langevin = fill(sqrt(2.0), dim)

# Simulation parameters
u0 = [0.0, 0.0, 0.0]
resolution = 10

Nsteps = 10000
dt = 0.001

# Integrate the same score-based system using DifferentialEquations
function drift_score_de(du, u, p, t)
    du .= Float64.(score_fn(u))
end

function diffusion_score_de(du, u, p, t)
    du .= sqrt(2.0)  # Same as σ_langevin
end

# Create the SDE problem for score-based generation
sde_prob_score = SDEProblem(drift_score_de, diffusion_score_de, u0, (0.0, Nsteps * dt), p)

# Time both score-based integrations together
@time begin
    traj2 = evolve(u0, dt, Nsteps, drift_score!, σ_langevin; timestepper=:rk4, resolution=resolution, seed=rand(UInt32), n_ens=100, boundary=(-3,3))
    sol_de_score = solve(sde_prob_score, EM(), dt=dt, saveat=resolution*dt)
end

# Convert solution to matrix format for comparison
traj2_de = hcat(sol_de_score.u...)

println("FastSDE score trajectory shape: ", size(traj2))
println("DifferentialEquations score trajectory shape: ", size(traj2_de))

# Extract the trajectory as a matrix (dim x time steps)
trj_clustered = traj2

#plot(trj_clustered[1,1:1:1000000], label="Xz", xlabel="Time", ylabel="Value", title="Trajectory X")

##
gr()

# Compute KDEs for FastSDE trajectories
kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])

kde_clustered_z = kde(trj_clustered[3,:])
kde_true_z = kde(obs[3,:])

# Compute KDEs for DifferentialEquations trajectories
kde_clustered_x_de = kde(traj2_de[1,:])
kde_clustered_y_de = kde(traj2_de[2,:])
kde_clustered_z_de = kde(traj2_de[3,:])

# Plot X component comparison
plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="FastSDE", xlabel="X", ylabel="Density", title="X Component PDF Comparison", xlims=(-3,3))
plt1 = Plots.plot!(kde_clustered_x_de.x, kde_clustered_x_de.density, label="DifferentialEquations", linewidth=2)
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", linewidth=2)

# Plot Y component comparison
plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="FastSDE", xlabel="Y", ylabel="Density", title="Y Component PDF Comparison", xlims=(-3,3))
plt2 = Plots.plot!(kde_clustered_y_de.x, kde_clustered_y_de.density, label="DifferentialEquations", linewidth=2)
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", linewidth=2)

# Plot Z component comparison
plt3 = Plots.plot(kde_clustered_z.x, kde_clustered_z.density, label="FastSDE", xlabel="Z", ylabel="Density", title="Z Component PDF Comparison", xlims=(-3,3))
plt3 = Plots.plot!(kde_clustered_z_de.x, kde_clustered_z_de.density, label="DifferentialEquations", linewidth=2)
plt3 = Plots.plot!(kde_true_z.x, kde_true_z.density, label="True", linewidth=2)

fig = Plots.plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 800))
#Plots.savefig(fig, "fastsde_vs_diffeq_vs_true_pdfs.png")
##

