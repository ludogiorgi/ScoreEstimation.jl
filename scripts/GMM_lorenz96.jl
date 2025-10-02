using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ScoreEstimation
using FastSDE
using KernelDensity
using HDF5
using Flux
using LinearAlgebra
using Statistics
using GLMakie

# Define Lorenz96 drift function
function drift!(du, u, t; F0=6.0, nu=1.0, c=10.0, b=10.0, Nk=4, Nj=10)
    c1 = c / b

    # Extract slow variables (first Nk)
    x_slow = u[1:Nk]

    # Extract and reshape fast variables
    x_fast = reshape(u[Nk+1:end], (Nj, Nk))'

    # Slow variable dynamics
    for k in 1:Nk
        im1 = mod(k - 2, Nk) + 1
        im2 = mod(k - 3, Nk) + 1
        ip1 = mod(k, Nk) + 1
        du[k] = -x_slow[im1] * (x_slow[im2] - x_slow[ip1]) - nu * x_slow[k] + F0 + c1 * sum(x_fast[k, :])
    end

    # Fast variable dynamics
    dy = zeros(Nk, Nj)
    for k in 1:Nk
        for j in 1:Nj
            jm1 = mod(j - 2, Nj) + 1
            jp1 = mod(j, Nj) + 1
            jp2 = mod(j + 1, Nj) + 1
            dy[k, j] = -c * b * x_fast[k, jp1] * (x_fast[k, jp2] - x_fast[k, jm1]) -
                       c * nu * x_fast[k, j] + c1 * x_slow[k]
        end
    end

    # Combine slow and fast derivatives
    du[Nk+1:end] .= vec(transpose(dy))
end

function diffusion!(du, u, t; noise=0.2)
    du .= noise
end

# Simulation parameters
dim = 4  # We'll only use the first 4 slow variables for analysis
total_dim = 4 + 4*10  # 4 slow + 4*10 fast variables
dt = 0.005
Nsteps = 4_000_000
resolution = 2
u0 = 0.01 .* randn(total_dim)

# Generate observations (full system)
obs_full = evolve(u0, dt, Nsteps, drift!, diffusion!;
                  timestepper=:euler, resolution=resolution, n_ens=100)

# Extract only slow variables for analysis
obs_nn = obs_full[1:dim, :]

# Normalization
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# True score function (normalized coordinates) - for slow variables only
function score_true_norm(x, M, S)
    # Reconstruct full state (slow + fast at equilibrium)
    x_phys = x .* S .+ M
    u_full = zeros(total_dim)
    u_full[1:dim] .= x_phys

    # Compute drift
    du_full = zeros(total_dim)
    drift!(du_full, u_full, 0.0)

    # Extract slow variable drift and transform
    return du_full[1:dim] .* S
end

# KGMM parameters
σ_value = 0.08
neurons = [128, 64]

# Train with KGMM preprocessing
@time nn_kgmm, losses_kgmm, _, div_kgmm, kgmm_res = ScoreEstimation.train(
    obs;
    preprocessing=true,
    σ=σ_value,
    neurons=neurons,
    n_epochs=2000,
    batch_size=32,
    lr=1e-3,
    use_gpu=false,
    verbose=true,
    kgmm_kwargs=(prob=0.0002, conv_param=0.002, i_max=100, show_progress=false),
    divergence=false,
    probes=1
)

# Score function from NN
score_nn(x) = -nn_kgmm(Float32.(x)) ./ Float32(σ_value)

# Generate trajectory using score-based model
σ_langevin = fill(1.0, dim)
dt_gen = 0.001
Nsteps_gen = 10_000_000
resolution_gen = 2

drift_score!(du, u, t) = (du .= Float64.(score_nn(u)))

trj_kgmm = evolve(zeros(dim), dt_gen, Nsteps_gen, drift_score!, σ_langevin;
                  timestepper=:euler, resolution=resolution_gen, n_ens=100)

# Compute average univariate PDF across all variables
kde_1 = kde(obs[1,:])
kde_2 = kde(obs[2,:])
kde_3 = kde(obs[3,:])
kde_4 = kde(obs[4,:])

kde_kgmm_1 = kde(trj_kgmm[1,:])
kde_kgmm_2 = kde(trj_kgmm[2,:])
kde_kgmm_3 = kde(trj_kgmm[3,:])
kde_kgmm_4 = kde(trj_kgmm[4,:])

# Average the PDFs
kde_true_x = ([kde_1.x...] .+ [kde_2.x...] .+ [kde_3.x...] .+ [kde_4.x...]) ./ 4
kde_true_y = (kde_1.density .+ kde_2.density .+ kde_3.density .+ kde_4.density) ./ 4

kde_kgmm_x = ([kde_kgmm_1.x...] .+ [kde_kgmm_2.x...] .+ [kde_kgmm_3.x...] .+ [kde_kgmm_4.x...]) ./ 4
kde_kgmm_y = (kde_kgmm_1.density .+ kde_kgmm_2.density .+ kde_kgmm_3.density .+ kde_kgmm_4.density) ./ 4

# Compute bivariate PDFs for consecutive variables
kde_true_12 = kde((obs[1,:], obs[2,:]))
kde_kgmm_12 = kde((trj_kgmm[1,:], trj_kgmm[2,:]))

kde_true_23 = kde((obs[2,:], obs[3,:]))
kde_kgmm_23 = kde((trj_kgmm[2,:], trj_kgmm[3,:]))

kde_true_34 = kde((obs[3,:], obs[4,:]))
kde_kgmm_34 = kde((trj_kgmm[3,:], trj_kgmm[4,:]))

kde_true_41 = kde((obs[4,:], obs[1,:]))
kde_kgmm_41 = kde((trj_kgmm[4,:], trj_kgmm[1,:]))

# Average consecutive PDFs
kde_true_consecutive_density = (kde_true_12.density + kde_true_23.density + kde_true_34.density + kde_true_41.density) ./ 4
kde_kgmm_consecutive_density = (kde_kgmm_12.density + kde_kgmm_23.density + kde_kgmm_34.density + kde_kgmm_41.density) ./ 4

# Use one grid for plotting
kde_consecutive_x = kde_true_12.x
kde_consecutive_y = kde_true_12.y

# Compute bivariate PDFs for skip-one variables
kde_true_13 = kde((obs[1,:], obs[3,:]))
kde_kgmm_13 = kde((trj_kgmm[1,:], trj_kgmm[3,:]))

kde_true_24 = kde((obs[2,:], obs[4,:]))
kde_kgmm_24 = kde((trj_kgmm[2,:], trj_kgmm[4,:]))

# Average skip-one PDFs
kde_true_skip_density = (kde_true_13.density + kde_true_24.density) ./ 2
kde_kgmm_skip_density = (kde_kgmm_13.density + kde_kgmm_24.density) ./ 2

# Use one grid for plotting
kde_skip_x = kde_true_13.x
kde_skip_y = kde_true_13.y

# Generate trajectory samples for time series (only first 2 components for clarity)
obs_trj = evolve(0.01 .* randn(total_dim), dt, 20000, drift!, diffusion!;
                 timestepper=:euler, resolution=2, n_ens=1)
obs_trj = ((obs_trj[1:dim,:] .- M) ./ S)[1:2,:]

score_trj = evolve(0.01 .* randn(dim), dt, 20000, drift_score!, σ_langevin;
                   timestepper=:euler, resolution=2, n_ens=1)[1:2,:]

# Save data to HDF5
function save_lorenz96_data(filename="data/GMM_data/lorenz96.h5")
    mkpath(dirname(filename))

    density_max = max(
        maximum(kde_true_consecutive_density),
        maximum(kde_kgmm_consecutive_density),
        maximum(kde_true_skip_density),
        maximum(kde_kgmm_skip_density)
    )

    h5open(filename, "w") do file
        # Univariate PDFs
        write(file, "kde_true_x", collect(kde_true_x))
        write(file, "kde_true_y", collect(kde_true_y))
        write(file, "kde_kgmm_x", collect(kde_kgmm_x))
        write(file, "kde_kgmm_y", collect(kde_kgmm_y))

        # Bivariate consecutive PDFs
        write(file, "kde_consecutive_x", collect(kde_consecutive_x))
        write(file, "kde_consecutive_y", collect(kde_consecutive_y))
        write(file, "kde_true_consecutive_density", kde_true_consecutive_density)
        write(file, "kde_kgmm_consecutive_density", kde_kgmm_consecutive_density)

        # Bivariate skip PDFs
        write(file, "kde_skip_x", collect(kde_skip_x))
        write(file, "kde_skip_y", collect(kde_skip_y))
        write(file, "kde_true_skip_density", kde_true_skip_density)
        write(file, "kde_kgmm_skip_density", kde_kgmm_skip_density)

        # Color scaling
        write(file, "density_max", density_max)

        # Metadata
        write(file, "dt", dt)
        write(file, "dim", dim)
        write(file, "σ_value", σ_value)

        # Sample data
        write(file, "obs_trj", obs_trj)
        write(file, "score_trj", score_trj)

        # Normalization parameters
        write(file, "M", M)
        write(file, "S", S)
    end

    println("Data saved to $filename")
end

save_lorenz96_data()

# Create figure
function create_lorenz96_figure(filename="data/GMM_data/lorenz96.h5")
    data = Dict()

    h5open(filename, "r") do file
        data["kde_true_x"] = read(file, "kde_true_x")
        data["kde_true_y"] = read(file, "kde_true_y")
        data["kde_kgmm_x"] = read(file, "kde_kgmm_x")
        data["kde_kgmm_y"] = read(file, "kde_kgmm_y")

        data["kde_consecutive_x"] = read(file, "kde_consecutive_x")
        data["kde_consecutive_y"] = read(file, "kde_consecutive_y")
        data["kde_true_consecutive_density"] = read(file, "kde_true_consecutive_density")
        data["kde_kgmm_consecutive_density"] = read(file, "kde_kgmm_consecutive_density")

        data["kde_skip_x"] = read(file, "kde_skip_x")
        data["kde_skip_y"] = read(file, "kde_skip_y")
        data["kde_true_skip_density"] = read(file, "kde_true_skip_density")
        data["kde_kgmm_skip_density"] = read(file, "kde_kgmm_skip_density")

        data["density_max"] = read(file, "density_max")
        data["dt"] = read(file, "dt")
        data["obs_trj"] = read(file, "obs_trj")
        data["score_trj"] = read(file, "score_trj")
    end

    println("Data loaded from $filename")

    # Create figure
    fig = GLMakie.Figure(size=(2250, 2250), fontsize=32)
    main_layout = fig[1:3, 1:3] = GLMakie.GridLayout()

    # Time series row
    time_panel = main_layout[1, 1:3] = GLMakie.GridLayout()

    # PDF rows
    left_panel = main_layout[2:3, 1] = GLMakie.GridLayout()
    right_panel = main_layout[2:3, 2:3] = GLMakie.GridLayout()

    # Time series axis
    time_ax1 = GLMakie.Axis(time_panel[1, 1:2],
                            xlabel="t", ylabel="x[k]", title="x[k] Time Series",
                            titlesize=36, xlabelsize=32, ylabelsize=32)

    # Plot time series
    n_points = 10000
    time_vector = (1:2:n_points) .* data["dt"] .* 2

    GLMakie.lines!(time_ax1, time_vector, data["obs_trj"][1, 1:2:n_points],
                   color=:red, linewidth=1, label="True")
    GLMakie.lines!(time_ax1, time_vector, data["score_trj"][1, 1:2:n_points],
                   color=:blue, linewidth=1, label="KGMM")

    # Univariate PDF axis
    univariate_ax = GLMakie.Axis(left_panel[1, 1],
                                 xlabel="x[k]", ylabel="PDF", title="Univariate PDF",
                                 titlesize=36, xlabelsize=32, ylabelsize=32)

    # Univariate plot
    GLMakie.lines!(univariate_ax, data["kde_true_x"], data["kde_true_y"],
                   color=:red, linewidth=2, label="True")
    GLMakie.lines!(univariate_ax, data["kde_kgmm_x"], data["kde_kgmm_y"],
                   color=:blue, linewidth=2, label="KGMM")
    GLMakie.axislegend(univariate_ax, position=:rt, framevisible=true,
                       bgcolor=:white, labelsize=32)

    # Heatmap axes
    heatmap_axes = [
        GLMakie.Axis(right_panel[1, 1], titlesize=36, xlabelsize=32, ylabelsize=32),
        GLMakie.Axis(right_panel[1, 2], titlesize=36, xlabelsize=32, ylabelsize=32),
        GLMakie.Axis(right_panel[2, 1], titlesize=36, xlabelsize=32, ylabelsize=32),
        GLMakie.Axis(right_panel[2, 2], titlesize=36, xlabelsize=32, ylabelsize=32)
    ]

    # Set titles and labels
    heatmap_axes[1].title = "True (x[k]-x[k+1]) PDF"
    heatmap_axes[2].title = "KGMM (x[k]-x[k+1]) PDF"
    heatmap_axes[3].title = "True (x[k]-x[k+2]) PDF"
    heatmap_axes[4].title = "KGMM (x[k]-x[k+2]) PDF"

    for ax in heatmap_axes
        ax.xlabel = "x[k]"
    end

    heatmap_axes[1].ylabel = "x[k+1]"
    heatmap_axes[2].ylabel = "x[k+1]"
    heatmap_axes[3].ylabel = "x[k+2]"
    heatmap_axes[4].ylabel = "x[k+2]"

    # Create heatmaps
    density_max = data["density_max"]

    GLMakie.heatmap!(heatmap_axes[1], data["kde_consecutive_x"], data["kde_consecutive_y"],
                     data["kde_true_consecutive_density"], colormap=:viridis, colorrange=(0, density_max))
    GLMakie.heatmap!(heatmap_axes[2], data["kde_consecutive_x"], data["kde_consecutive_y"],
                     data["kde_kgmm_consecutive_density"], colormap=:viridis, colorrange=(0, density_max))
    GLMakie.heatmap!(heatmap_axes[3], data["kde_skip_x"], data["kde_skip_y"],
                     data["kde_true_skip_density"], colormap=:viridis, colorrange=(0, density_max))
    GLMakie.heatmap!(heatmap_axes[4], data["kde_skip_x"], data["kde_skip_y"],
                     data["kde_kgmm_skip_density"], colormap=:viridis, colorrange=(0, density_max))

    # Colorbar
    GLMakie.Colorbar(fig[2:3, 4], limits=(0, density_max), colormap=:viridis,
                     ticklabelsize=32, width=30)

    # Adjust layout
    GLMakie.colgap!(main_layout, 15)
    GLMakie.rowgap!(main_layout, 15)
    GLMakie.colgap!(right_panel, 15)
    GLMakie.rowgap!(right_panel, 15)
    GLMakie.colgap!(time_panel, 15)

    # Balance row sizes
    GLMakie.rowsize!(main_layout, 1, GLMakie.Relative(0.3))
    GLMakie.rowsize!(main_layout, 2, GLMakie.Relative(0.35))
    GLMakie.rowsize!(main_layout, 3, GLMakie.Relative(0.35))

    mkpath("figures/GMM_figures")
    GLMakie.save("figures/GMM_figures/lorenz96.png", fig)

    return fig
end

fig = create_lorenz96_figure()
println("Figure saved to figures/GMM_figures/lorenz96.png")
