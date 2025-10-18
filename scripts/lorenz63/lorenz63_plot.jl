using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CairoMakie
using CairoMakie: Figure, Axis, Legend, Theme, set_theme!, lines!, scatter!, scatterlines!, axislegend, save, GridLayout
using HDF5
using Colors
using Statistics

const L63_ROOT = @__DIR__
const L63_FIGURES_DIR = joinpath(L63_ROOT, "figures")
const L63_PUB_DIR = joinpath(L63_FIGURES_DIR, "publication")
const L63_DATA_DIR = joinpath(L63_ROOT, "data")

set_theme!(Theme(
    fontsize = 16,
    font = "CMU Serif",
    linewidth = 2.5,
    markersize = 10,
    size = (1000, 700),
    backgroundcolor = :white,
    Axis = (
        backgroundcolor = :white,
        xgridcolor = :gray90,
        ygridcolor = :gray90,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xlabelsize = 16,
        ylabelsize = 16,
        titlesize = 18,
        spinewidth = 1.5,
    )
))

color_nopreproc = colorant"#d62728"
color_preproc = colorant"#2ca02c"
color_obs = colorant"#1f77b4"
color_gen = colorant"#ff7f0e"

mkpath(L63_PUB_DIR)

# -----------------------------------------------------------------------------
# Performance comparison figure
# -----------------------------------------------------------------------------

@info "Loading performance data..."
perf_path = joinpath(L63_DATA_DIR, "lorenz63_performances.h5")
h5open(perf_path, "r") do file
    train_times_no_pre = read(file, "results_no_pre_time")
    rel_ent_no_pre = read(file, "results_no_pre_relent")
    train_times_pre = read(file, "results_pre_time")
    rel_ent_pre = read(file, "results_pre_relent")
    dims = size(rel_ent_no_pre, 1)
    axes_labels = ["x", "y", "z"]

    # Compute average KL divergence across dimensions
    avg_rel_ent_no_pre = vec(mean(rel_ent_no_pre, dims=1))
    avg_rel_ent_pre = vec(mean(rel_ent_pre, dims=1))

    fig1 = Figure(size=(800, 600))
    ax = Axis(fig1[1, 1],
        xlabel = "Training time (s)",
        ylabel = "Average relative entropy",
        title = "Lorenz-63",
        xscale = log10,
        yscale = log10,
    )

    scatterlines!(ax, train_times_no_pre, avg_rel_ent_no_pre;
        color = color_nopreproc,
        marker = :circle,
        linewidth = 2,
        markersize = 12,
        label = "No KGMM")
    scatterlines!(ax, train_times_pre, avg_rel_ent_pre;
        color = color_preproc,
        marker = :diamond,
        linewidth = 2,
        markersize = 12,
        label = "KGMM")

    Legend(fig1[2, 1], ax,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        framevisible = true,
        labelsize = 14)

    save(joinpath(L63_PUB_DIR, "lorenz63_performance_comparison.png"), fig1, px_per_unit=2)
    save(joinpath(L63_PUB_DIR, "lorenz63_performance_comparison.pdf"), fig1)
    @info "Saved performance figure"
end

# -----------------------------------------------------------------------------
# Detailed analysis figure
# -----------------------------------------------------------------------------

@info "Loading compute analysis..."
compute_path = joinpath(L63_DATA_DIR, "lorenz63_compute.h5")
h5open(compute_path, "r") do file
    time_axis = read(file, "trajectory_time")
    trajectory_obs = read(file, "trajectory_obs")
    trajectory_gen = read(file, "trajectory_gen")
    rel_ent = read(file, "relative_entropy")
    kde_obs_x_x = read(file, "kde_obs_x_x")
    kde_obs_x_density = read(file, "kde_obs_x_density")
    kde_gen_x_x = read(file, "kde_gen_x_x")
    kde_gen_x_density = read(file, "kde_gen_x_density")
    kde_obs_y_x = read(file, "kde_obs_y_x")
    kde_obs_y_density = read(file, "kde_obs_y_density")
    kde_gen_y_x = read(file, "kde_gen_y_x")
    kde_gen_y_density = read(file, "kde_gen_y_density")
    kde_obs_z_x = read(file, "kde_obs_z_x")
    kde_obs_z_density = read(file, "kde_obs_z_density")
    kde_gen_z_x = read(file, "kde_gen_z_x")
    kde_gen_z_density = read(file, "kde_gen_z_density")

    fig2 = Figure(size=(1200, 1800))

    ax1 = Axis(fig2[1, 1],
        xlabel = "Time",
        ylabel = "State x",
        title = "Trajectory comparison (x-component)",
    )
    lines!(ax1, time_axis, trajectory_obs[1, :]; color = color_obs, linewidth = 2, label = "Observed")
    lines!(ax1, time_axis, trajectory_gen[1, :]; color = color_gen, linewidth = 2, linestyle = :dash, label = "NN Langevin")
    axislegend(ax1, position = :rb, framevisible = true, labelsize = 14)

    ax2 = Axis(fig2[2, 1],
        xlabel = "Time",
        ylabel = "State y",
        title = "Trajectory comparison (y-component)",
    )
    lines!(ax2, time_axis, trajectory_obs[2, :]; color = color_obs, linewidth = 2, label = "Observed")
    lines!(ax2, time_axis, trajectory_gen[2, :]; color = color_gen, linewidth = 2, linestyle = :dash, label = "NN Langevin")
    axislegend(ax2, position = :rb, framevisible = true, labelsize = 14)

    ax3 = Axis(fig2[3, 1],
        xlabel = "Time",
        ylabel = "State z",
        title = "Trajectory comparison (z-component)",
    )
    lines!(ax3, time_axis, trajectory_obs[3, :]; color = color_obs, linewidth = 2, label = "Observed")
    lines!(ax3, time_axis, trajectory_gen[3, :]; color = color_gen, linewidth = 2, linestyle = :dash, label = "NN Langevin")
    axislegend(ax3, position = :rb, framevisible = true, labelsize = 14)

    pdf_grid = fig2[4, 1] = GridLayout(1, 3)

    ax_pdf_x = Axis(pdf_grid[1, 1],
        xlabel = "x (norm.)",
        ylabel = "PDF",
        title = "Normalized PDF (x)",
    )
    lines!(ax_pdf_x, kde_obs_x_x, kde_obs_x_density; color = color_obs, linewidth = 2, label = "Observed")
    lines!(ax_pdf_x, kde_gen_x_x, kde_gen_x_density; color = color_gen, linewidth = 2, linestyle = :dash, label = "NN Langevin")
    axislegend(ax_pdf_x, position = :lt, framevisible = true, labelsize = 14)

    ax_pdf_y = Axis(pdf_grid[1, 2],
        xlabel = "y (norm.)",
        ylabel = "PDF",
        title = "Normalized PDF (y)",
    )
    lines!(ax_pdf_y, kde_obs_y_x, kde_obs_y_density; color = color_obs, linewidth = 2)
    lines!(ax_pdf_y, kde_gen_y_x, kde_gen_y_density; color = color_gen, linewidth = 2, linestyle = :dash)

    ax_pdf_z = Axis(pdf_grid[1, 3],
        xlabel = "z (norm.)",
        ylabel = "PDF",
        title = "Normalized PDF (z)",
    )
    lines!(ax_pdf_z, kde_obs_z_x, kde_obs_z_density; color = color_obs, linewidth = 2)
    lines!(ax_pdf_z, kde_gen_z_x, kde_gen_z_density; color = color_gen, linewidth = 2, linestyle = :dash)

    ax4 = Axis(fig2[5, 1],
        xlabel = "x",
        ylabel = "z",
        title = "Phase portrait (x – z)\nRelative entropy = $(round.(rel_ent, digits=4))",
    )
    scatter!(ax4, trajectory_obs[1, :], trajectory_obs[3, :];
        color = (color_obs, 0.6),
        markersize = 5,
        label = "Observed")
    scatter!(ax4, trajectory_gen[1, :], trajectory_gen[3, :];
        color = (color_gen, 0.6),
        markersize = 5,
        label = "NN Langevin")
    axislegend(ax4, position = :rt, framevisible = true, labelsize = 14)

    save(joinpath(L63_PUB_DIR, "lorenz63_detailed_analysis.png"), fig2, px_per_unit=2)
    save(joinpath(L63_PUB_DIR, "lorenz63_detailed_analysis.pdf"), fig2)
    @info "Saved detailed analysis figure"
end

@info "All publication-ready Lorenz-63 figures generated"
