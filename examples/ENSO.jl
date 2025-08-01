using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
# Add required packages if not already in the environment
# Pkg.add("DifferentialEquations")
# Pkg.add("StatsBase")
# Pkg.add("Plots")
# Pkg.add("LinearAlgebra")

using ScoreEstimation
using DifferentialEquations
using StatsBase
using Plots
using LinearAlgebra
using Flux
using KernelDensity

# Define parameters as a named tuple
p = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

# Drift function with signature f(u, p, t)
function F(u, p, t)
    F1 = -p.dᵤ * u[1] - p.wᵤ * u[2] + u[3]
    F2 = -p.dᵤ * u[2] + p.wᵤ * u[1]
    F3 = -p.dₜ * u[3]
    return [F1, F2, F3]
end

# Diffusion function with signature g(u, p, t) - return vector for diagonal noise
function sigma(u, p, t)
    sigma1 = p.σ₁
    sigma2 = p.σ₂
    sigma3 = 1.5 * (tanh(u[1]) + 1)
    return [sigma1, sigma2, sigma3]  # Vector for diagonal diffusion
end

dim = 3
dt = 0.01
Nsteps = 10000000
tspan = (0.0, Nsteps * dt)
u0 = [0.0, 0.0, 0.0]

# Define the SDE problem with parameters p
prob = SDEProblem(F, sigma, u0, tspan, p; noise_rate_prototype=nothing)  # Explicitly set for diagonal noise

# Solve the SDE using Euler-Maruyama method, saving every 'resolution' steps
resolution = 10
sol = solve(prob, EM(), dt=dt, saveat=resolution*dt)

# Extract the trajectory as a matrix (dim x time steps)
obs_nn = hcat(sol.u...)

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
############################ CLUSTERING ####################

obs_uncorr = obs[:, 1:1:end]
normalization = false
σ_value = 0.05

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0001, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

plotly()
targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], centers[3,:], marker_z=targets_norm, color=:viridis)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 200, 16, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##
#################### SAMPLES GENERATION ####################

# Assuming score_clustered and score_qG are defined elsewhere
score_clustered_xt(x, t) = score_clustered(x)

# Define parameters (adjust dim if needed)
p = (dim=3, )  # Add other params if necessary

# Drift function with signature f(u, p, t)
function drift(u, p, t)
    return score_clustered_xt(u, t)
end

# Diffusion function with signature g(u, p, t) - vector for diagonal noise
function sigma(u, p, t)
    return √2 * ones(p.dim)  # [√2, √2, √2] for dim=3
end

# Simulation parameters
dt = 0.0005
Nsteps = 100000000  # For testing, reduce to 1e6 or similar
tspan = (0.0, Nsteps * dt)
u0 = [0.0, 0.0, 0.0]
resolution = 10

# Define the SDE problem with parameters p
prob = SDEProblem(drift, sigma, u0, tspan, p; noise_rate_prototype=nothing)

# Solve the SDE using fast Euler-Maruyama (disable adaptive for speed)
sol = solve(prob, EM(), dt=dt, saveat=resolution*dt)  # No callback for max speed; add if needed

# Extract the trajectory as a matrix (dim x time steps)
trj_clustered = hcat(sol.u...)

#plot(trj_clustered[1,1:1:1000000], label="X", xlabel="Time", ylabel="Value", title="Trajectory X")

##
gr()
kde_clustered_x = kde(trj_clustered[1,1:1000000])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,1:10000000])
kde_true_y = kde(obs[2,:])

kde_clustered_z = kde(trj_clustered[3,1:10000000])
kde_true_z = kde(obs[3,:])

plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")

plt3 = Plots.plot(kde_clustered_z.x, kde_clustered_z.density, label="Observed", xlabel="Z", ylabel="Density", title="Observed PDF")
plt3 = Plots.plot!(kde_true_z.x, kde_true_z.density, label="True", xlabel="Z", ylabel="Density", title="True PDF")

fig = Plots.plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 800))
#Plots.savefig(fig, "clustered_vs_true_pdfs.png")
##

