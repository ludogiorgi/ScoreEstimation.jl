using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ScoreEstimation
using FastSDE
using LinearAlgebra
using Statistics
using KernelDensity
using HDF5
using Flux
using GLMakie

# Define the drift function
function drift!(du, u, t; A1=1.0, A2=1.2, B1=0.6, B2=0.3)
    ∇U1 = 2 * (u[1] + A1) * (u[1] - A1)^2 + 2 * (u[1] - A1) * (u[1] + A1)^2 + B1
    ∇U2 = 2 * (u[2] + A2) * (u[2] - A2)^2 + 2 * (u[2] - A2) * (u[2] + A2)^2 + B2
    du[1] = -∇U1
    du[2] = -∇U2
end

function diffusion!(du, u, t; noise=1.0)
    du[1] = noise
    du[2] = noise
end

# Simulation parameters
dim = 2
dt = 0.05
Nsteps = 100_000
resolution = 100
u0 = [0.0, 0.0]

# Generate observations
obs_nn = evolve(u0, dt, Nsteps, drift!, diffusion!;
                timestepper=:euler, resolution=resolution, n_ens=100)

# Normalization
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# True score function (normalized coordinates)
function score_true_norm(x, M, S)
    # Transform to physical coordinates
    x_phys = x .* S .+ M
    # Compute drift at physical coordinates
    du = zeros(2)
    drift!(du, x_phys, 0.0)
    # Transform score back to normalized coordinates
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
resolution_gen = 10

drift_score!(du, u, t) = (du .= Float64.(score_nn(u)))

trj_kgmm = evolve(zeros(dim), dt_gen, Nsteps_gen, drift_score!, σ_langevin;
                  timestepper=:euler, resolution=resolution_gen, n_ens=100)

# Compute KDEs
kde_true_x = kde(obs[1,:])
kde_kgmm_x = kde(trj_kgmm[1,:])

kde_true_y = kde(obs[2,:])
kde_kgmm_y = kde(trj_kgmm[2,:])

kde_true_xy = kde((obs[1,:], obs[2,:]))
kde_kgmm_xy = kde((trj_kgmm[1,:], trj_kgmm[2,:]))

# Generate observation and score trajectories for time series
obs_trj = evolve(zeros(dim), dt, 10000, drift!, diffusion!;
                 timestepper=:euler, n_ens=1)
obs_trj = (obs_trj .- M) ./ S

score_trj = evolve(zeros(dim), dt_gen, 100000, drift_score!, σ_langevin;
                   timestepper=:euler, resolution=10, n_ens=1)

# Create vector field data
n_grid = 50
d_grid = 1/10
c_grid = [((n_grid+1)*d_grid)/2, ((n_grid+1)*d_grid)/2]

x = range(-c_grid[1], stop=c_grid[1], length=n_grid)
y = range(-c_grid[2], stop=c_grid[2], length=n_grid)

u_true = zeros(n_grid, n_grid)
v_true = zeros(n_grid, n_grid)
u_kgmm = zeros(n_grid, n_grid)
v_kgmm = zeros(n_grid, n_grid)

for i in 1:n_grid
    for j in 1:n_grid
        score_t = score_true_norm([x[i], y[j]], M, S)
        u_true[j, i] = score_t[1]
        v_true[j, i] = score_t[2]

        score_k = score_nn([x[i], y[j]])
        u_kgmm[j, i] = score_k[1]
        v_kgmm[j, i] = score_k[2]
    end
end

# Save data to HDF5
mkpath("data/GMM_data")

h5open("data/GMM_data/2D_potential.h5", "w") do file
    write(file, "dt", dt)
    write(file, "dim", dim)
    write(file, "Nsteps", Nsteps)

    # Vector field grid
    write(file, "x", collect(x))
    write(file, "y", collect(y))

    # Force field data
    write(file, "u_true", u_true)
    write(file, "v_true", v_true)
    write(file, "u_kgmm", u_kgmm)
    write(file, "v_kgmm", v_kgmm)

    # KDE data
    write(file, "kde_true_x_x", collect(kde_true_x.x))
    write(file, "kde_true_x_density", collect(kde_true_x.density))
    write(file, "kde_kgmm_x_x", collect(kde_kgmm_x.x))
    write(file, "kde_kgmm_x_density", collect(kde_kgmm_x.density))

    write(file, "kde_true_y_x", collect(kde_true_y.x))
    write(file, "kde_true_y_density", collect(kde_true_y.density))
    write(file, "kde_kgmm_y_x", collect(kde_kgmm_y.x))
    write(file, "kde_kgmm_y_density", collect(kde_kgmm_y.density))

    write(file, "kde_true_xy_x", collect(kde_true_xy.x))
    write(file, "kde_true_xy_y", collect(kde_true_xy.y))
    write(file, "kde_true_xy_density", kde_true_xy.density)

    write(file, "kde_kgmm_xy_x", collect(kde_kgmm_xy.x))
    write(file, "kde_kgmm_xy_y", collect(kde_kgmm_xy.y))
    write(file, "kde_kgmm_xy_density", kde_kgmm_xy.density)

    # Trajectory samples
    write(file, "obs_trj", obs_trj)
    write(file, "score_trj", score_trj)

    # Statistics
    write(file, "M", M)
    write(file, "S", S)
end

println("Data saved to data/GMM_data/2D_potential.h5")

# Create figure
function create_figure(filename="data/GMM_data/2D_potential.h5")
    data = Dict()
    h5open(filename, "r") do file
        data["x"] = read(file, "x")
        data["y"] = read(file, "y")
        data["u_true"] = read(file, "u_true")
        data["v_true"] = read(file, "v_true")
        data["u_kgmm"] = read(file, "u_kgmm")
        data["v_kgmm"] = read(file, "v_kgmm")
        data["kde_true_x_x"] = read(file, "kde_true_x_x")
        data["kde_true_x_density"] = read(file, "kde_true_x_density")
        data["kde_kgmm_x_x"] = read(file, "kde_kgmm_x_x")
        data["kde_kgmm_x_density"] = read(file, "kde_kgmm_x_density")
        data["kde_true_y_x"] = read(file, "kde_true_y_x")
        data["kde_true_y_density"] = read(file, "kde_true_y_density")
        data["kde_kgmm_y_x"] = read(file, "kde_kgmm_y_x")
        data["kde_kgmm_y_density"] = read(file, "kde_kgmm_y_density")
        data["kde_true_xy_x"] = read(file, "kde_true_xy_x")
        data["kde_true_xy_y"] = read(file, "kde_true_xy_y")
        data["kde_true_xy_density"] = read(file, "kde_true_xy_density")
        data["kde_kgmm_xy_x"] = read(file, "kde_kgmm_xy_x")
        data["kde_kgmm_xy_y"] = read(file, "kde_kgmm_xy_y")
        data["kde_kgmm_xy_density"] = read(file, "kde_kgmm_xy_density")
        data["obs_trj"] = read(file, "obs_trj")
        data["score_trj"] = read(file, "score_trj")
        data["dt"] = read(file, "dt")
    end

    x = data["x"]
    y = data["y"]
    u_true = data["u_true"]
    v_true = data["v_true"]
    u_kgmm = data["u_kgmm"]
    v_kgmm = data["v_kgmm"]
    obs_sample = data["obs_trj"]
    score_sample = data["score_trj"]
    dt = data["dt"]

    # Create figure
    fig = GLMakie.Figure(size=(2500, 1400), fontsize=28)
    grid = fig[1:3, 1:8] = GLMakie.GridLayout(3, 8)

    # Time series plots
    ax_time_x = GLMakie.Axis(grid[1, 1:4], xlabel="t", ylabel="x", title="x Time Series",
                             titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_time_y = GLMakie.Axis(grid[1, 5:8], xlabel="t", ylabel="y", title="y Time Series",
                             titlesize=36, xlabelsize=32, ylabelsize=32)

    # Vector field axes
    ax_vf_true = GLMakie.Axis(grid[2, 1:2], xlabel="x", ylabel="y", title="True Score",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_vf_kgmm = GLMakie.Axis(grid[3, 1:2], xlabel="x", ylabel="y", title="KGMM Score",
                              titlesize=36, xlabelsize=32, ylabelsize=32)

    # PDF axes
    ax_pdf_x = GLMakie.Axis(grid[2, 4:5], xlabel="x", ylabel="PDF", title="Univariate x PDFs",
                            titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_pdf_y = GLMakie.Axis(grid[3, 4:5], xlabel="y", ylabel="PDF", title="Univariate y PDFs",
                            titlesize=36, xlabelsize=32, ylabelsize=32)

    # Bivariate PDF axes
    ax_true_xy = GLMakie.Axis(grid[2, 6:7], xlabel="x", ylabel="y", title="Bivariate True PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)
    ax_kgmm_xy = GLMakie.Axis(grid[3, 6:7], xlabel="x", ylabel="y", title="Bivariate KGMM PDF",
                              titlesize=36, xlabelsize=32, ylabelsize=32)

    # Plot time series
    time_subset = 1:1:min(1000, size(obs_sample, 2))
    time_points = collect(time_subset) .* dt

    GLMakie.lines!(ax_time_x, time_points, obs_sample[1, time_subset], color=:red, linewidth=1, label="True")
    GLMakie.lines!(ax_time_x, time_points, score_sample[1, time_subset], color=:blue, linewidth=1, label="KGMM")
    GLMakie.lines!(ax_time_y, time_points, obs_sample[2, time_subset], color=:red, linewidth=1)
    GLMakie.lines!(ax_time_y, time_points, score_sample[2, time_subset], color=:blue, linewidth=1)

    # Vector fields
    x_points = repeat(x, outer=length(y))
    y_points = repeat(y, inner=length(x))
    u_true_flat = vec(u_true')
    v_true_flat = vec(v_true')
    u_kgmm_flat = vec(u_kgmm')
    v_kgmm_flat = vec(v_kgmm')

    mag_true_flat = sqrt.(u_true_flat.^2 .+ v_true_flat.^2)
    mag_kgmm_flat = sqrt.(u_kgmm_flat.^2 .+ v_kgmm_flat.^2)

    scale = 1.0
    u_true_norm = u_true_flat ./ max.(mag_true_flat, 1e-10) .* scale
    v_true_norm = v_true_flat ./ max.(mag_true_flat, 1e-10) .* scale
    u_kgmm_norm = u_kgmm_flat ./ max.(mag_kgmm_flat, 1e-10) .* scale
    v_kgmm_norm = v_kgmm_flat ./ max.(mag_kgmm_flat, 1e-10) .* scale

    vf_vmax = maximum(mag_kgmm_flat)

    GLMakie.arrows!(ax_vf_true, x_points, y_points, u_true_norm, v_true_norm,
                    arrowsize=1, linewidth=1, color=mag_true_flat, colormap=:viridis, colorrange=(0, vf_vmax))
    GLMakie.arrows!(ax_vf_kgmm, x_points, y_points, u_kgmm_norm, v_kgmm_norm,
                    arrowsize=1, linewidth=1, color=mag_kgmm_flat, colormap=:viridis, colorrange=(0, vf_vmax))

    # Univariate PDFs
    GLMakie.lines!(ax_pdf_x, data["kde_true_x_x"], data["kde_true_x_density"], color=:red, linewidth=2, label="True")
    GLMakie.lines!(ax_pdf_x, data["kde_kgmm_x_x"], data["kde_kgmm_x_density"], color=:blue, linewidth=2, label="KGMM")
    GLMakie.axislegend(ax_pdf_x, position=:lt, labelsize=32)

    GLMakie.lines!(ax_pdf_y, data["kde_true_y_x"], data["kde_true_y_density"], color=:red, linewidth=2, label="True")
    GLMakie.lines!(ax_pdf_y, data["kde_kgmm_y_x"], data["kde_kgmm_y_density"], color=:blue, linewidth=2, label="KGMM")

    # Bivariate PDFs
    pdf_vmax = max(maximum(data["kde_true_xy_density"]), maximum(data["kde_kgmm_xy_density"]))

    GLMakie.heatmap!(ax_true_xy, data["kde_true_xy_x"], data["kde_true_xy_y"], data["kde_true_xy_density"],
                     colormap=:viridis, colorrange=(0, pdf_vmax))
    GLMakie.heatmap!(ax_kgmm_xy, data["kde_kgmm_xy_x"], data["kde_kgmm_xy_y"], data["kde_kgmm_xy_density"],
                     colormap=:viridis, colorrange=(0, pdf_vmax))

    # Colorbars
    GLMakie.Colorbar(grid[2:3, 3], colormap=:viridis, limits=(0, vf_vmax), labelsize=32, vertical=true, width=20)
    GLMakie.Colorbar(grid[2:3, 8], colormap=:viridis, limits=(0, pdf_vmax), labelsize=32, vertical=true, width=20)

    # Adjust column widths
    GLMakie.colsize!(grid, 3, GLMakie.Relative(0.05))
    GLMakie.colsize!(grid, 8, GLMakie.Relative(0.05))
    GLMakie.colgap!(grid, 10)
    GLMakie.rowgap!(grid, 15)

    mkpath("figures/GMM_figures")
    GLMakie.save("figures/GMM_figures/2D_potential.png", fig)

    return fig
end

fig = create_figure()
println("Figure saved to figures/GMM_figures/2D_potential.png")
