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

# Define Lorenz63 drift function
function drift!(du, u, t; σ=10.0, ρ=28.0, β=8/3)
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

function diffusion!(du, u, t; noise=5.0)
    du[1] = noise
    du[2] = noise
    du[3] = noise
end

# Simulation parameters
dim = 3
dt = 0.01
Nsteps = 1_000_000
resolution = 10
u0 = [1.0, 1.5, 1.8]

# Generate observations
obs_nn = evolve(u0, dt, Nsteps, drift!, diffusion!;
                timestepper=:euler, resolution=resolution, n_ens=100)

# Normalization
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# True score function (normalized coordinates)
function score_true_norm(x, M, S)
    x_phys = x .* S .+ M
    du = zeros(3)
    drift!(du, x_phys, 0.0)
    return du .* S
end

# KGMM parameters
σ_value = 0.05
neurons = [128, 64]

# Train with KGMM preprocessing
@time nn_kgmm, losses_kgmm, _, div_kgmm, kgmm_res = ScoreEstimation.train(
    obs;
    preprocessing=true,
    σ=σ_value,
    neurons=neurons,
    n_epochs=1000,
    batch_size=32,
    lr=1e-3,
    use_gpu=false,
    verbose=true,
    kgmm_kwargs=(prob=0.001, conv_param=0.005, i_max=100, show_progress=false),
    divergence=false,
    probes=1
)

# Score function from NN
score_nn(x) = -nn_kgmm(Float32.(x)) ./ Float32(σ_value)

# Generate trajectory using score-based model
σ_langevin = fill(1.0, dim)
dt_gen = 0.005
Nsteps_gen = 2_000_000
resolution_gen = 2

drift_score!(du, u, t) = (du .= Float64.(score_nn(u)))

trj_kgmm = evolve(zeros(dim), dt_gen, Nsteps_gen, drift_score!, σ_langevin;
                  timestepper=:euler, resolution=resolution_gen, n_ens=100)

# Compute univariate KDEs
kde_true_x = kde(obs[1,:])
kde_kgmm_x = kde(trj_kgmm[1,:])

kde_true_y = kde(obs[2,:])
kde_kgmm_y = kde(trj_kgmm[2,:])

kde_true_z = kde(obs[3,:])
kde_kgmm_z = kde(trj_kgmm[3,:])

# Compute bivariate KDEs
kde_true_xy = kde((obs[1,:], obs[2,:]))
kde_kgmm_xy = kde((trj_kgmm[1,:], trj_kgmm[2,:]))

kde_true_xz = kde((obs[1,:], obs[3,:]))
kde_kgmm_xz = kde((trj_kgmm[1,:], trj_kgmm[3,:]))

kde_true_yz = kde((obs[2,:], obs[3,:]))
kde_kgmm_yz = kde((trj_kgmm[2,:], trj_kgmm[3,:]))

# Generate trajectory samples for time series
obs_trj = evolve(zeros(dim), dt, 10000, drift!, diffusion!;
                 timestepper=:euler, n_ens=1)
obs_trj = (obs_trj .- M) ./ S

score_trj = evolve(zeros(dim), dt_gen, 100000, drift_score!, σ_langevin;
                   timestepper=:euler, resolution=10, n_ens=1)

# Save data to HDF5
function save_lorenz63_data(filename="data/GMM_data/lorenz63.h5")
    mkpath(dirname(filename))

    h5open(filename, "w") do file
        write(file, "dt", dt)
        write(file, "dim", dim)
        write(file, "Nsteps", Nsteps)

        # Model parameters
        write(file, "M", M)
        write(file, "S", S)

        # Univariate KDE data
        write(file, "kde_true_x_x", collect(kde_true_x.x))
        write(file, "kde_true_x_density", collect(kde_true_x.density))
        write(file, "kde_kgmm_x_x", collect(kde_kgmm_x.x))
        write(file, "kde_kgmm_x_density", collect(kde_kgmm_x.density))

        write(file, "kde_true_y_x", collect(kde_true_y.x))
        write(file, "kde_true_y_density", collect(kde_true_y.density))
        write(file, "kde_kgmm_y_x", collect(kde_kgmm_y.x))
        write(file, "kde_kgmm_y_density", collect(kde_kgmm_y.density))

        write(file, "kde_true_z_x", collect(kde_true_z.x))
        write(file, "kde_true_z_density", collect(kde_true_z.density))
        write(file, "kde_kgmm_z_x", collect(kde_kgmm_z.x))
        write(file, "kde_kgmm_z_density", collect(kde_kgmm_z.density))

        # Bivariate KDE data
        write(file, "kde_true_xy_x", collect(kde_true_xy.x))
        write(file, "kde_true_xy_y", collect(kde_true_xy.y))
        write(file, "kde_true_xy_density", kde_true_xy.density)

        write(file, "kde_kgmm_xy_x", collect(kde_kgmm_xy.x))
        write(file, "kde_kgmm_xy_y", collect(kde_kgmm_xy.y))
        write(file, "kde_kgmm_xy_density", kde_kgmm_xy.density)

        write(file, "kde_true_xz_x", collect(kde_true_xz.x))
        write(file, "kde_true_xz_y", collect(kde_true_xz.y))
        write(file, "kde_true_xz_density", kde_true_xz.density)

        write(file, "kde_kgmm_xz_x", collect(kde_kgmm_xz.x))
        write(file, "kde_kgmm_xz_y", collect(kde_kgmm_xz.y))
        write(file, "kde_kgmm_xz_density", kde_kgmm_xz.density)

        write(file, "kde_true_yz_x", collect(kde_true_yz.x))
        write(file, "kde_true_yz_y", collect(kde_true_yz.y))
        write(file, "kde_true_yz_density", kde_true_yz.density)

        write(file, "kde_kgmm_yz_x", collect(kde_kgmm_yz.x))
        write(file, "kde_kgmm_yz_y", collect(kde_kgmm_yz.y))
        write(file, "kde_kgmm_yz_density", kde_kgmm_yz.density)

        # Sample data
        write(file, "obs_trj", obs_trj)
        write(file, "score_trj", score_trj)
    end

    println("Data saved to $filename")
end

save_lorenz63_data()

# Create figure
function create_lorenz63_figure(filename="data/GMM_data/lorenz63.h5")
    data = Dict()

    h5open(filename, "r") do file
        data["dt"] = read(file, "dt")
        data["dim"] = read(file, "dim")

        data["kde_true_x_x"] = read(file, "kde_true_x_x")
        data["kde_true_x_density"] = read(file, "kde_true_x_density")
        data["kde_kgmm_x_x"] = read(file, "kde_kgmm_x_x")
        data["kde_kgmm_x_density"] = read(file, "kde_kgmm_x_density")

        data["kde_true_y_x"] = read(file, "kde_true_y_x")
        data["kde_true_y_density"] = read(file, "kde_true_y_density")
        data["kde_kgmm_y_x"] = read(file, "kde_kgmm_y_x")
        data["kde_kgmm_y_density"] = read(file, "kde_kgmm_y_density")

        data["kde_true_z_x"] = read(file, "kde_true_z_x")
        data["kde_true_z_density"] = read(file, "kde_true_z_density")
        data["kde_kgmm_z_x"] = read(file, "kde_kgmm_z_x")
        data["kde_kgmm_z_density"] = read(file, "kde_kgmm_z_density")

        data["kde_true_xy_x"] = read(file, "kde_true_xy_x")
        data["kde_true_xy_y"] = read(file, "kde_true_xy_y")
        data["kde_true_xy_density"] = read(file, "kde_true_xy_density")
        data["kde_kgmm_xy_x"] = read(file, "kde_kgmm_xy_x")
        data["kde_kgmm_xy_y"] = read(file, "kde_kgmm_xy_y")
        data["kde_kgmm_xy_density"] = read(file, "kde_kgmm_xy_density")

        data["kde_true_xz_x"] = read(file, "kde_true_xz_x")
        data["kde_true_xz_y"] = read(file, "kde_true_xz_y")
        data["kde_true_xz_density"] = read(file, "kde_true_xz_density")
        data["kde_kgmm_xz_x"] = read(file, "kde_kgmm_xz_x")
        data["kde_kgmm_xz_y"] = read(file, "kde_kgmm_xz_y")
        data["kde_kgmm_xz_density"] = read(file, "kde_kgmm_xz_density")

        data["kde_true_yz_x"] = read(file, "kde_true_yz_x")
        data["kde_true_yz_y"] = read(file, "kde_true_yz_y")
        data["kde_true_yz_density"] = read(file, "kde_true_yz_density")
        data["kde_kgmm_yz_x"] = read(file, "kde_kgmm_yz_x")
        data["kde_kgmm_yz_y"] = read(file, "kde_kgmm_yz_y")
        data["kde_kgmm_yz_density"] = read(file, "kde_kgmm_yz_density")

        data["obs_trj"] = read(file, "obs_trj")
        data["score_trj"] = read(file, "score_trj")
    end

    println("Data loaded from $filename")

    # Create figure
    fig = GLMakie.Figure(size=(2250, 1800), fontsize=28)
    grid = fig[1:4, 1:3] = GLMakie.GridLayout(4, 3)

    # Time series plots
    ax_time_x = GLMakie.Axis(grid[1, 1], xlabel="t", ylabel="x", title="x Time Series",
                             titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_time_y = GLMakie.Axis(grid[1, 2], xlabel="t", ylabel="y", title="y Time Series",
                             titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_time_z = GLMakie.Axis(grid[1, 3], xlabel="t", ylabel="z", title="z Time Series",
                             titlesize=36, xlabelsize=32, ylabelsize=32)

    # Univariate PDF axes
    ax_pdf_x = GLMakie.Axis(grid[2, 1], xlabel="x", ylabel="PDF", title="Univariate x PDF",
                            titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_pdf_y = GLMakie.Axis(grid[3, 1], xlabel="y", ylabel="PDF", title="Univariate y PDF",
                            titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_pdf_z = GLMakie.Axis(grid[4, 1], xlabel="z", ylabel="PDF", title="Univariate z PDF",
                            titlesize=36, xlabelsize=32, ylabelsize=32)

    # True bivariate PDF axes
    ax_true_xy = GLMakie.Axis(grid[2, 2], xlabel="x", ylabel="y", title="True (x,y) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_true_xz = GLMakie.Axis(grid[3, 2], xlabel="x", ylabel="z", title="True (x,z) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_true_yz = GLMakie.Axis(grid[4, 2], xlabel="y", ylabel="z", title="True (y,z) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)

    # KGMM bivariate PDF axes
    ax_kgmm_xy = GLMakie.Axis(grid[2, 3], xlabel="x", ylabel="y", title="KGMM (x,y) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_kgmm_xz = GLMakie.Axis(grid[3, 3], xlabel="x", ylabel="z", title="KGMM (x,z) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_kgmm_yz = GLMakie.Axis(grid[4, 3], xlabel="y", ylabel="z", title="KGMM (y,z) PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)

    # Plot time series
    obs_trj = data["obs_trj"]
    score_trj = data["score_trj"]
    dt = data["dt"]
    n_time_points = min(1000, size(obs_trj, 2))
    time_points = collect(1:n_time_points) .* dt

    GLMakie.lines!(ax_time_x, time_points, obs_trj[1, 1:n_time_points], color=:red, linewidth=1, label="True")
    GLMakie.lines!(ax_time_x, time_points, score_trj[1, 1:n_time_points], color=:blue, linewidth=1, label="KGMM")

    GLMakie.lines!(ax_time_y, time_points, obs_trj[2, 1:n_time_points], color=:red, linewidth=1)
    GLMakie.lines!(ax_time_y, time_points, score_trj[2, 1:n_time_points], color=:blue, linewidth=1)

    GLMakie.lines!(ax_time_z, time_points, obs_trj[3, 1:n_time_points], color=:red, linewidth=1)
    GLMakie.lines!(ax_time_z, time_points, score_trj[3, 1:n_time_points], color=:blue, linewidth=1)

    # Univariate PDFs
    GLMakie.lines!(ax_pdf_x, data["kde_true_x_x"], data["kde_true_x_density"], color=:red, linewidth=2, label="True")
    GLMakie.lines!(ax_pdf_x, data["kde_kgmm_x_x"], data["kde_kgmm_x_density"], color=:blue, linewidth=2, label="KGMM")
    GLMakie.axislegend(ax_pdf_x, position=:lt, labelsize=32)

    GLMakie.lines!(ax_pdf_y, data["kde_true_y_x"], data["kde_true_y_density"], color=:red, linewidth=2, label="True")
    GLMakie.lines!(ax_pdf_y, data["kde_kgmm_y_x"], data["kde_kgmm_y_density"], color=:blue, linewidth=2, label="KGMM")

    GLMakie.lines!(ax_pdf_z, data["kde_true_z_x"], data["kde_true_z_density"], color=:red, linewidth=2, label="True")
    GLMakie.lines!(ax_pdf_z, data["kde_kgmm_z_x"], data["kde_kgmm_z_density"], color=:blue, linewidth=2, label="KGMM")

    # Color scaling for bivariate PDFs
    pdf_min = min(
        minimum(data["kde_true_xy_density"]), minimum(data["kde_kgmm_xy_density"]),
        minimum(data["kde_true_xz_density"]), minimum(data["kde_kgmm_xz_density"]),
        minimum(data["kde_true_yz_density"]), minimum(data["kde_kgmm_yz_density"])
    )
    pdf_max = max(
        maximum(data["kde_true_xy_density"]), maximum(data["kde_kgmm_xy_density"]),
        maximum(data["kde_true_xz_density"]), maximum(data["kde_kgmm_xz_density"]),
        maximum(data["kde_true_yz_density"]), maximum(data["kde_kgmm_yz_density"])
    )

    # Bivariate PDFs
    GLMakie.heatmap!(ax_true_xy, data["kde_true_xy_x"], data["kde_true_xy_y"], data["kde_true_xy_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))
    GLMakie.heatmap!(ax_true_xz, data["kde_true_xz_x"], data["kde_true_xz_y"], data["kde_true_xz_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))
    GLMakie.heatmap!(ax_true_yz, data["kde_true_yz_x"], data["kde_true_yz_y"], data["kde_true_yz_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))

    GLMakie.heatmap!(ax_kgmm_xy, data["kde_kgmm_xy_x"], data["kde_kgmm_xy_y"], data["kde_kgmm_xy_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))
    GLMakie.heatmap!(ax_kgmm_xz, data["kde_kgmm_xz_x"], data["kde_kgmm_xz_y"], data["kde_kgmm_xz_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))
    GLMakie.heatmap!(ax_kgmm_yz, data["kde_kgmm_yz_x"], data["kde_kgmm_yz_y"], data["kde_kgmm_yz_density"],
                     colormap=:viridis, colorrange=(pdf_min, pdf_max))

    # Colorbar
    GLMakie.Colorbar(fig[2:4, 4], colormap=:viridis, limits=(pdf_min, pdf_max),
                     labelsize=32, vertical=true, width=30)

    mkpath("figures/GMM_figures")
    GLMakie.save("figures/GMM_figures/lorenz63.png", fig)

    return fig
end

fig = create_lorenz63_figure()
println("Figure saved to figures/GMM_figures/lorenz63.png")
