using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CairoMakie
using CairoMakie: Figure, Axis, Legend, Theme, set_theme!, scatter!, lines!, axislegend, save
using HDF5
using Colors
const REDUCED_ROOT = @__DIR__
const REDUCED_FIGURES_DIR = joinpath(REDUCED_ROOT, "figures")
const REDUCED_PUB_DIR = joinpath(REDUCED_FIGURES_DIR, "publication")
const REDUCED_DATA_DIR = joinpath(REDUCED_ROOT, "data")

# Set publication-ready theme
set_theme!(Theme(
    fontsize = 16,
    font = "CMU Serif",
    linewidth = 2.5,
    markersize = 12,
    size = (800, 600),
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

# Define colors for consistent styling
color_true = colorant"#1f77b4"      # blue
color_nn = colorant"#ff7f0e"        # orange
color_emp = colorant"#7f7f7f"       # gray
color_nopreproc = colorant"#d62728" # red
color_preproc = colorant"#2ca02c"   # green

mkpath(REDUCED_PUB_DIR)

# ============================================================================
# FIGURE 1: Performance Comparison (preprocessing true vs false)
# ============================================================================

@info "Loading performance comparison data..."
h5open(joinpath(REDUCED_DATA_DIR, "reduced_performances.h5"), "r") do file
    train_times_no_pre = read(file, "results_no_pre_time")
    rel_ent_no_pre = read(file, "results_no_pre_relent")
    train_times_pre = read(file, "results_pre_time")
    rel_ent_pre = read(file, "results_pre_relent")

    @info "Creating performance comparison figure..."
    fig1 = Figure(size=(900, 700))
    ax1 = Axis(fig1[1, 1],
        xlabel = "Training time (s)",
        ylabel = "Relative entropy",
        title = "Score Estimation Performance:\nPreprocessing Comparison",
        xscale = log10,
        yscale = log10,
    )

    # Plot data
    scatter!(ax1, train_times_no_pre, rel_ent_no_pre,
        marker = :circle,
        markersize = 14,
        color = color_nopreproc,
        strokewidth = 1.5,
        strokecolor = :black,
    label = "Without KGMM")

    lines!(ax1, train_times_no_pre, rel_ent_no_pre,
        color = (color_nopreproc, 0.5),
        linewidth = 2,
        linestyle = :solid)

    scatter!(ax1, train_times_pre, rel_ent_pre,
        marker = :rect,
        markersize = 14,
        color = color_preproc,
        strokewidth = 1.5,
        strokecolor = :black,
    label = "With KGMM")

    lines!(ax1, train_times_pre, rel_ent_pre,
        color = (color_preproc, 0.5),
        linewidth = 2,
        linestyle = :solid)

    # Add legend
    Legend(fig1[2, 1], ax1, orientation = :horizontal, valign = :bottom, halign = :center,
        framevisible = true, labelsize = 14)

    # Save figure
    save(joinpath(REDUCED_PUB_DIR, "reduced_performance_comparison.png"), fig1, px_per_unit=2)
    save(joinpath(REDUCED_PUB_DIR, "reduced_performance_comparison.pdf"), fig1)

    @info "Saved performance comparison figure"
end

# ============================================================================
# FIGURE 2: Detailed Analysis (3 subplots)
# ============================================================================

@info "Loading compute analysis data..."
h5open(joinpath(REDUCED_DATA_DIR, "reduced_compute.h5"), "r") do file
    # Read trajectory data
    trajectory_obs = read(file, "trajectory_obs")
    trajectory_nn = read(file, "trajectory_nn")

    # Read score function data
    x_grid = read(file, "x_grid")
    score_true = read(file, "score_true")
    score_nn = read(file, "score_nn")

    # Read PDF data
    pdf_true_x = read(file, "pdf_true_x")
    pdf_true_density = read(file, "pdf_true_density")
    pdf_nn_x = read(file, "pdf_nn_x")
    pdf_nn_density = read(file, "pdf_nn_density")
    pdf_emp_x = read(file, "pdf_emp_x")
    pdf_emp_density = read(file, "pdf_emp_density")

    # Read metadata
    n_clusters = read(file, "n_clusters")
    rel_ent = read(file, "relative_entropy")

    @info "Creating detailed analysis figure..."
    fig2 = Figure(size=(1000, 1400))

    # Subplot 1: Trajectory comparison
    ax1 = Axis(fig2[1, 1],
        xlabel = "Time step",
        ylabel = "State x",
        title = "Trajectory Comparison",
    )

    n_points = min(2000, length(trajectory_obs), length(trajectory_nn))
    lines!(ax1, 1:n_points, trajectory_obs[1:n_points],
        color = color_true,
        linewidth = 2,
        alpha = 0.8,
        label = "Observed")

    lines!(ax1, 1:n_points, trajectory_nn[1:n_points],
        color = color_nn,
        linewidth = 2,
        alpha = 0.8,
        linestyle = :dash,
        label = "NN score")

    axislegend(ax1, position = :rt, framevisible = true, labelsize = 14)

    # Subplot 2: Score function comparison
    ax2 = Axis(fig2[2, 1],
        xlabel = "State x",
        ylabel = "Score s(x) = ∂ₓlog p(x)",
        title = "Score Function Comparison",
    )

    lines!(ax2, x_grid, score_true,
        color = color_true,
        linewidth = 2.5,
        label = "Analytic score")

    lines!(ax2, x_grid, score_nn,
        color = color_nn,
        linewidth = 2.5,
        linestyle = :dash,
        label = "NN score")

    axislegend(ax2, position = :lt, framevisible = true, labelsize = 14)

    # Subplot 3: PDF comparison
    ax3 = Axis(fig2[3, 1],
        xlabel = "State x",
        ylabel = "Probability density p(x)",
        title = "Stationary PDF Comparison\n(Clusters = $n_clusters, D_KL = $(round(rel_ent, digits=4)))",
    )

    lines!(ax3, pdf_true_x, pdf_true_density,
        color = color_true,
        linewidth = 2.5,
        label = "Reference (analytic)")

    lines!(ax3, pdf_nn_x, pdf_nn_density,
        color = color_nn,
        linewidth = 2.5,
        linestyle = :dash,
        label = "NN score")

    lines!(ax3, pdf_emp_x, pdf_emp_density,
        color = color_emp,
        linewidth = 2,
        linestyle = :dot,
        label = "Empirical (obs)")

    axislegend(ax3, position = :rt, framevisible = true, labelsize = 14)

    # Save figure
    save(joinpath(REDUCED_PUB_DIR, "reduced_detailed_analysis.png"), fig2, px_per_unit=2)
    save(joinpath(REDUCED_PUB_DIR, "reduced_detailed_analysis.pdf"), fig2)

    @info "Saved detailed analysis figure"
end

@info "All publication-ready figures generated successfully!"
@info "Figures saved in: $(REDUCED_PUB_DIR)"
println("\nGenerated files:")
println("  - reduced_performance_comparison.png/.pdf")
println("  - reduced_detailed_analysis.png/.pdf")
