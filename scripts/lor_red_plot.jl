using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using CairoMakie
using CairoMakie: Figure, Axis, Legend, scatterlines!, save
using HDF5
using Colors
using Statistics

# Define directory paths
const LORENZ63_DATA_DIR = joinpath(@__DIR__, "lorenz63", "data")
const REDUCED_DATA_DIR = joinpath(@__DIR__, "reduced", "data")
const FIGURES_DIR = @__DIR__

# Set publication-ready theme
set_theme!(Theme(
    fontsize = 18,
    font = "CMU Serif",
    linewidth = 2.5,
    markersize = 12,
    backgroundcolor = :white,
    Axis = (
        backgroundcolor = :white,
        xgridcolor = :gray90,
        ygridcolor = :gray90,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
        titlesize = 20,
        spinewidth = 1.5,
    )
))

# Define colors for consistent styling
color_nopreproc = colorant"#d62728"  # red
color_preproc = colorant"#2ca02c"    # green

@info "Loading data files..."

# Load reduced model performance data
reduced_data = h5open(joinpath(REDUCED_DATA_DIR, "reduced_performances.h5"), "r") do file
    (
        train_times_no_pre = read(file, "results_no_pre_time"),
        rel_ent_no_pre = read(file, "results_no_pre_relent"),
        train_times_pre = read(file, "results_pre_time"),
        rel_ent_pre = read(file, "results_pre_relent")
    )
end

# Load Lorenz-63 performance data
lorenz63_data = h5open(joinpath(LORENZ63_DATA_DIR, "lorenz63_performances.h5"), "r") do file
    rel_ent_no_pre = read(file, "results_no_pre_relent")
    rel_ent_pre = read(file, "results_pre_relent")

    # Compute average KL divergence across dimensions (x, y, z)
    avg_rel_ent_no_pre = vec(mean(rel_ent_no_pre, dims=1))
    avg_rel_ent_pre = vec(mean(rel_ent_pre, dims=1))

    (
        train_times_no_pre = read(file, "results_no_pre_time"),
        rel_ent_no_pre = avg_rel_ent_no_pre,
        train_times_pre = read(file, "results_pre_time"),
        rel_ent_pre = avg_rel_ent_pre
    )
end

@info "Creating combined figure..."

# Create figure with two panels stacked vertically
fig = Figure(size=(1000, 1200), backgroundcolor=:white)

# Top panel: Reduced model
ax1 = Axis(fig[1, 1],
    xlabel = "Training time (s)",
    ylabel = "Relative entropy",
    title = "1D Reduced Model",
    yscale = log10,
    titlealign = :left,
    titlegap = 12,
)

scatterlines!(ax1, reduced_data.train_times_no_pre, reduced_data.rel_ent_no_pre;
    color = color_nopreproc,
    marker = :circle,
    linewidth = 2.5,
    markersize = 14,
    strokewidth = 1,
    strokecolor = :white,
    label = "No KGMM")

scatterlines!(ax1, reduced_data.train_times_pre, reduced_data.rel_ent_pre;
    color = color_preproc,
    marker = :diamond,
    linewidth = 2.5,
    markersize = 14,
    strokewidth = 1,
    strokecolor = :white,
    label = "KGMM")

# Bottom panel: Lorenz-63
ax2 = Axis(fig[2, 1],
    xlabel = "Training time (s)",
    ylabel = "Average relative entropy",
    title = "Lorenz-63 System",
    yscale = log10,
    titlealign = :left,
    titlegap = 12,
)

scatterlines!(ax2, lorenz63_data.train_times_no_pre, lorenz63_data.rel_ent_no_pre;
    color = color_nopreproc,
    marker = :circle,
    linewidth = 2.5,
    markersize = 14,
    strokewidth = 1,
    strokecolor = :white,
    label = "No KGMM")

scatterlines!(ax2, lorenz63_data.train_times_pre, lorenz63_data.rel_ent_pre;
    color = color_preproc,
    marker = :diamond,
    linewidth = 2.5,
    markersize = 14,
    strokewidth = 1,
    strokecolor = :white,
    label = "KGMM")

# Create a single legend at the bottom of the figure, outside both panels
Legend(fig[3, 1], ax2,
    orientation = :horizontal,
    tellwidth = false,
    tellheight = true,
    framevisible = true,
    labelsize = 18,
    framewidth = 1.5,
    padding = (10, 10, 10, 10),
    halign = :center,
    valign = :top,
    margin = (0, 0, 10, 0),
)

# Adjust row sizes to make plots equally sized and legend compact
rowsize!(fig.layout, 1, Auto(1.0))
rowsize!(fig.layout, 2, Auto(1.0))
rowsize!(fig.layout, 3, Auto(0.15))

# Adjust row gaps for better spacing
rowgap!(fig.layout, 1, 20)
rowgap!(fig.layout, 2, 15)

# Save figure in multiple formats
output_png = joinpath(FIGURES_DIR, "combined_performance_comparison.png")
output_pdf = joinpath(FIGURES_DIR, "combined_performance_comparison.pdf")

save(output_png, fig, px_per_unit=2)
save(output_pdf, fig)

@info "Figure saved successfully!"
@info "PNG: $output_png"
@info "PDF: $output_pdf"

println("\nGenerated publication-ready figure:")
println("  - $(basename(output_png))")
println("  - $(basename(output_pdf))")
