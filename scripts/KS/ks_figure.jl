using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed; continuing with existing environment" error=err
end

using CairoMakie
using CairoMakie: DataAspect, rowgap!, colgap!
using HDF5
using Statistics

const KS_ROOT = @__DIR__
const KS_RUNS_DIR = joinpath(KS_ROOT, "runs")

const FIG_FONT = get(ENV, "KS_FIG_FONT", "DejaVu Sans")
CairoMakie.activate!()
set_theme!(Theme(
    font = FIG_FONT,
    fontsize = 32,
    Figure = (figure_padding = 12,),
    Axis = (
        titlegap = 20,
        titlesize = 38,
        xlabelsize = 34,
        ylabelsize = 34,
        xticklabelsize = 28,
        yticklabelsize = 28,
        xgridvisible = false,
        ygridvisible = false,
        spinewidth = 2.0,
    ),
    Colorbar = (
        ticklabelsize = 28,
        labelsize = 32,
        width = 28,
    ),
    Legend = (
        labelsize = 30,
        framevisible = false,
        padding = (4, 4, 4, 4),
    ),
    Lines = (
        linewidth = 5,
    ),
))

"""
    load_run_data(run_name::String) -> NamedTuple

Load all necessary data from a run's output.h5 file.
"""
function load_run_data(run_name::String)
    h5_path = joinpath(KS_RUNS_DIR, run_name, "output.h5")
    @info "Loading run data" run=run_name path=h5_path
    
    data = h5open(h5_path, "r") do h5
        Xn_emp = read(h5, "Xn")
        X_gen_n = read(h5, "X_gen_n")
        mean_field = read(h5, "mean_field")
        selected_indices = Int.(read(h5, "selected_indices"))
        reduced_means = read(h5, "reduced_means")
        reduced_stds = read(h5, "reduced_stds")
        kde_emp_x = read(h5, "kde_emp_x")
        kde_emp_density = read(h5, "kde_emp_density")
        kde_gen_x = read(h5, "kde_gen_x")
        kde_gen_density = read(h5, "kde_gen_density")
        data_stride = Int(read(h5, "data_stride"))
        mode_stride = Int(read(h5, "mode_stride"))
        return (
            Xn_emp=Xn_emp,
            X_gen_n=X_gen_n,
            mean_field=mean_field,
            selected_indices=selected_indices,
            reduced_means=reduced_means,
            reduced_stds=reduced_stds,
            kde_emp_x=kde_emp_x,
            kde_emp_density=kde_emp_density,
            kde_gen_x=kde_gen_x,
            kde_gen_density=kde_gen_density,
            data_stride=data_stride,
            mode_stride=mode_stride
        )
    end
    
    return data
end

function unnormalize_data(Xn::AbstractMatrix, means::AbstractVector, stds::AbstractVector)
    means_mat = reshape(means, :, 1)
    stds_mat = reshape(stds, :, 1)
    return (Xn .* stds_mat) .+ means_mat
end

function assemble_physical_subset(X::AbstractMatrix, mean_field::AbstractVector, indices::AbstractVector{Int})
    subset_mean = reshape(mean_field[indices], :, 1)
    return subset_mean .+ X
end

struct PDFEstimate
    x::Vector{Float64}
    density::Vector{Float64}
end

struct BivariatePDFEstimate
    x::Vector{Float64}
    y::Vector{Float64}
    density::Matrix{Float64}
end

function compute_symmetric_value_range(datasets; padding_fraction::Float64=0.02, max_samples::Int=500_000)
    samples = Float64[]
    sizehint!(samples, min(max_samples, 50_000))
    for data in datasets
        vec_data = vec(Float64.(data))
        total_len = length(vec_data)
        remaining = max_samples - length(samples)
        if remaining <= 0
            break
        elseif total_len <= remaining
            append!(samples, vec_data)
        else
            stride = cld(total_len, remaining)
            append!(samples, vec_data[1:stride:end])
        end
    end
    filter!(isfinite, samples)
    isempty(samples) && return (-1.0, 1.0)
    abs_max = maximum(abs.(samples))
    if !isfinite(abs_max) || abs_max == 0
        abs_max = 1.0
    end
    abs_max *= (1 + max(padding_fraction, 0.0))
    return (-abs_max, abs_max)
end

function compute_averaged_pdfs(data::AbstractMatrix;
                               value_range::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    n_dims, n_times = size(data)
    data_min, data_max = value_range === nothing ? extrema(data) : value_range
    nbins_uni = 120
    bin_edges = range(data_min, data_max; length=nbins_uni + 1)
    bin_width = (data_max - data_min) / nbins_uni
    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins_uni))
    avg_density = zeros(Float64, nbins_uni)
    for i in 1:n_dims
        samples = vec(Float64.(data[i, :]))
        counts = zeros(Float64, nbins_uni)
        for val in samples
            if isfinite(val) && data_min <= val <= data_max
                bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins_uni)
                counts[bin_idx] += 1
            end
        end
        counts ./= (n_times * bin_width)
        avg_density .+= counts
    end
    avg_density ./= n_dims
    avg_univariate = PDFEstimate(x_centers, avg_density)
    distances = [1, 2, 3]
    avg_bivariates = BivariatePDFEstimate[]
    nbins_biv = 80
    x_biv_edges = range(data_min, data_max; length=nbins_biv + 1)
    x_biv_width = (data_max - data_min) / nbins_biv
    x_biv_centers = collect(range(data_min + x_biv_width/2, data_max - x_biv_width/2; length=nbins_biv))
    y_biv_centers = x_biv_centers
    for dist in distances
        avg_biv_density = zeros(Float64, nbins_biv, nbins_biv)
        for i in 1:n_dims
            j = mod1(i + dist, n_dims)
            samples_x = vec(Float64.(data[i, :]))
            samples_y = vec(Float64.(data[j, :]))
            density = zeros(Float64, nbins_biv, nbins_biv)
            for t in 1:n_times
                xv = samples_x[t]
                yv = samples_y[t]
                if isfinite(xv) && isfinite(yv) &&
                   data_min <= xv <= data_max &&
                   data_min <= yv <= data_max
                    xi = clamp(searchsortedlast(x_biv_edges, xv), 1, nbins_biv)
                    yi = clamp(searchsortedlast(x_biv_edges, yv), 1, nbins_biv)
                    density[yi, xi] += 1
                end
            end
            area = x_biv_width * x_biv_width
            if area > 0
                density ./= (n_times * area)
            end
            avg_biv_density .+= density
        end
        avg_biv_density ./= n_dims
        push!(avg_bivariates, BivariatePDFEstimate(x_biv_centers, y_biv_centers, avg_biv_density))
    end
    return avg_univariate, avg_bivariates
end

const KS_REFERENCE_DATAFILE = joinpath(KS_ROOT, "data", "ks_old_reduced.hdf5")

function load_reference_dataset(; max_snapshots::Int=1000)
    @info "Loading reference KS data" path=KS_REFERENCE_DATAFILE max_snapshots
    raw_matrix = h5open(KS_REFERENCE_DATAFILE, "r") do file
        haskey(file, "u") || error("Dataset key `u` not found in $(KS_REFERENCE_DATAFILE)")
        dset = file["u"]
        data = Array(dset)
        close(dset)
        data
    end
    raw_matrix = convert(Matrix{Float64}, raw_matrix)
    n_time = size(raw_matrix, 2)
    n_snapshots = max_snapshots > 0 ? min(max_snapshots, n_time) : n_time
    time_indices = collect(1:n_snapshots)
    subset = raw_matrix[:, time_indices]
    return subset, time_indices
end

function select_reference_modes(reference_data::AbstractMatrix, latent_dim::Int)
    total_modes = size(reference_data, 1)
    total_modes >= latent_dim || error("Reference data has $(total_modes) modes, cannot select $(latent_dim)")
    stride = cld(total_modes, latent_dim)
    indices = collect(1:stride:total_modes)
    if length(indices) > latent_dim
        indices = indices[1:latent_dim]
    elseif length(indices) < latent_dim
        # Fallback to evenly spaced indices if cld under-samples (should not trigger for nominal configurations)
        indices = round.(Int, range(1, total_modes; length=latent_dim))
    end
    return reference_data[indices, :], indices, stride
end

function build_comparison_figure(runs::Vector{String}, strides::Vector{Int})
    @assert length(runs) == 3 "Expected exactly 3 runs"
    @assert length(strides) == 3 "Expected exactly 3 strides"
    
    fig = Figure(size=(2400, 3000), fontsize=24)
    rowgap!(fig.layout, 16)
    colgap!(fig.layout, 20)
    
    # Grid rows retain contextual meaning via axis titles and labels; no separate row captions.
    
    all_data = NamedTuple[]
    orig_min = Inf
    orig_max = -Inf

    reference_data_full, reference_time_indices = load_reference_dataset(max_snapshots=1000)
    
    for (col_idx, run_name) in enumerate(runs)
        stride = strides[col_idx]
        data = load_run_data(run_name)
        X_emp = unnormalize_data(data.Xn_emp, data.reduced_means, data.reduced_stds)
        X_gen = unnormalize_data(data.X_gen_n, data.reduced_means, data.reduced_stds)
        empirical_subset = assemble_physical_subset(X_emp, data.mean_field, data.selected_indices)
        generated_subset = assemble_physical_subset(X_gen, data.mean_field, data.selected_indices)
        reference_subset, _, reference_stride = select_reference_modes(reference_data_full, size(empirical_subset, 1))
        orig_min = min(orig_min, minimum(reference_subset))
        orig_max = max(orig_max, maximum(reference_subset))
        
        push!(all_data, (run_name=run_name, stride=stride, data=data,
                         empirical_subset=empirical_subset, generated_subset=generated_subset,
                         kde_emp_x=data.kde_emp_x, kde_emp_density=data.kde_emp_density,
                         kde_gen_x=data.kde_gen_x, kde_gen_density=data.kde_gen_density,
                         reference_subset=reference_subset, reference_time_indices=reference_time_indices,
                         reference_stride=reference_stride, selected_indices=data.selected_indices))
    end
    
    datasets_for_range = Any[]
    for run_data in all_data
        push!(datasets_for_range, run_data.empirical_subset)
        push!(datasets_for_range, run_data.generated_subset)
    end
    value_range = compute_symmetric_value_range(datasets_for_range; padding_fraction=0.05)
    
    pdf_max_values = [maximum(run_data.kde_emp_density) for run_data in all_data]
    append!(pdf_max_values, [maximum(run_data.kde_gen_density) for run_data in all_data])
    max_pdf = maximum(pdf_max_values)
    pdf_ylim = max_pdf > 0 ? (0.0, max_pdf * 1.05) : (0.0, 1.0)
    
    all_biv_emp = []
    all_biv_gen = []
    for run_data in all_data
        avg_uni_emp, avg_biv_emp = compute_averaged_pdfs(run_data.empirical_subset; value_range=value_range)
        avg_uni_gen, avg_biv_gen = compute_averaged_pdfs(run_data.generated_subset; value_range=value_range)
        push!(all_biv_emp, avg_biv_emp)
        push!(all_biv_gen, avg_biv_gen)
    end
    
    biv_max_dist1 = maximum([maximum(all_biv_emp[i][1].density) for i in 1:3] ∪ 
                             [maximum(all_biv_gen[i][1].density) for i in 1:3])
    biv_max_dist2 = maximum([maximum(all_biv_emp[i][2].density) for i in 1:3] ∪ 
                             [maximum(all_biv_gen[i][2].density) for i in 1:3])
    
    hm_orig = nothing
    hm_biv_emp_d1 = nothing
    hm_biv_emp_d2 = nothing
    line_emp_handle = nothing
    line_gen_handle = nothing
    
    for (col_idx, (run_name, stride, data, empirical_subset, generated_subset,
                   kde_emp_x, kde_emp_density, kde_gen_x, kde_gen_density,
                   reference_subset, reference_time_indices, reference_stride, selected_indices)) in enumerate(all_data)
        @info "Processing run $(col_idx)/3" run=run_name stride=stride
        
        avg_biv_emp = all_biv_emp[col_idx]
        avg_biv_gen = all_biv_gen[col_idx]
        
        latent_dim = size(empirical_subset, 1)
        @info "Row 1 subsampling" col=col_idx N=latent_dim ref_stride=reference_stride
        
        col_title = "N=$latent_dim (stride=$reference_stride)"
        ax1 = Axis(fig[1, col_idx];
                   xlabel="Time index",
                   ylabel="Space index",
                   title=col_title)
        if col_idx != 1
            ax1.ylabelvisible = false
        end
        hm_tmp = heatmap!(ax1,
                          collect(reference_time_indices),
                          collect(1:latent_dim),
                          permutedims(reference_subset);
                          colormap=:plasma,
                          colorrange=(orig_min, orig_max),
                          interpolate=true)
        ax1.yticks = (1:latent_dim, string.(1:latent_dim))
        if col_idx == 1
            hm_orig = hm_tmp
        end
        
        ax2 = Axis(fig[2, col_idx];
                   xlabel="Field value",
                   ylabel="Density")
        if col_idx != 1
            ax2.ylabelvisible = false
        end
        # Use colors from the plasma colormap for consistency
        plasma_cmap = cgrad(:plasma)
        emp_line = lines!(ax2, kde_emp_x, kde_emp_density; color=plasma_cmap[0.2], linewidth=5)
        gen_line = lines!(ax2, kde_gen_x, kde_gen_density; color=plasma_cmap[0.8], linewidth=5)
        xlims!(ax2, value_range...)
        ylims!(ax2, pdf_ylim...)
        if line_emp_handle === nothing
            line_emp_handle = emp_line
            line_gen_handle = gen_line
        end
        
        ax3 = Axis(fig[3, col_idx];
                   xlabel="x_i",
                   ylabel="x_{i+1}",
                   aspect=DataAspect())
        if col_idx != 1
            ax3.ylabelvisible = false
        end
        hm_tmp = heatmap!(ax3, avg_biv_emp[1].x, avg_biv_emp[1].y, avg_biv_emp[1].density;
                          colormap=:plasma,
                          colorrange=(0, biv_max_dist1),
                          interpolate=true)
        xlims!(ax3, value_range...)
        ylims!(ax3, value_range...)
        if col_idx == 1
            hm_biv_emp_d1 = hm_tmp
        end
        
        ax4 = Axis(fig[4, col_idx];
                   xlabel="x_i",
                   ylabel="x_{i+1}",
                   aspect=DataAspect())
        if col_idx != 1
            ax4.ylabelvisible = false
        end
        heatmap!(ax4, avg_biv_gen[1].x, avg_biv_gen[1].y, avg_biv_gen[1].density;
                 colormap=:plasma,
                 colorrange=(0, biv_max_dist1),
                 interpolate=true)
        xlims!(ax4, value_range...)
        ylims!(ax4, value_range...)
        
        ax5 = Axis(fig[5, col_idx];
                   xlabel="x_i",
                   ylabel="x_{i+2}",
                   aspect=DataAspect())
        if col_idx != 1
            ax5.ylabelvisible = false
        end
        hm_tmp = heatmap!(ax5, avg_biv_emp[2].x, avg_biv_emp[2].y, avg_biv_emp[2].density;
                          colormap=:plasma,
                          colorrange=(0, biv_max_dist2),
                          interpolate=true)
        xlims!(ax5, value_range...)
        ylims!(ax5, value_range...)
        if col_idx == 1
            hm_biv_emp_d2 = hm_tmp
        end
        
        ax6 = Axis(fig[6, col_idx];
                   xlabel="x_i",
                   ylabel="x_{i+2}",
                   aspect=DataAspect())
        if col_idx != 1
            ax6.ylabelvisible = false
        end
        heatmap!(ax6, avg_biv_gen[2].x, avg_biv_gen[2].y, avg_biv_gen[2].density;
                 colormap=:plasma,
                 colorrange=(0, biv_max_dist2),
                 interpolate=true)
        xlims!(ax6, value_range...)
        ylims!(ax6, value_range...)
    end
    
    Colorbar(fig[1, 4], hm_orig, label="Field value")
    if line_emp_handle !== nothing && line_gen_handle !== nothing
        Legend(fig[2, 4], [line_emp_handle, line_gen_handle],
               ["Empirical", "Generated"]; orientation=:vertical)
    end
    Colorbar(fig[3:4, 4], hm_biv_emp_d1, label="Density  (delta = 1)")
    Colorbar(fig[5:6, 4], hm_biv_emp_d2, label="Density  (delta = 2)")
    
    return fig
end

@info "Starting KS multi-run comparison figure generation"
runs = ["run_003", "run_001", "run_002"]
strides = [8, 4, 2]
fig = build_comparison_figure(runs, strides)

output_path = joinpath(KS_ROOT, "figure_ks.png")
CairoMakie.save(output_path, fig)
@info "Saved comparison figure" path=output_path
@info "Done!"
