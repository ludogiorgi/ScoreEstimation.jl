ENV["JULIA_PKG_PRECOMPILE_AUTO"] = get(ENV, "JULIA_PKG_PRECOMPILE_AUTO", "0")
ENV["JULIA_MAKIE_BACKEND"] = get(ENV, "JULIA_MAKIE_BACKEND", "cairomakie")

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed; continuing with existing environment" error=err
end

using CairoMakie
using HDF5
using Printf
using Statistics
using TOML

CairoMakie.activate!()

struct KSConfig
    stride::Int
    max_snapshots::Int
    n_modes::Int
    seed::Int
    sigma::Float64
    neurons::Vector{Int}
    n_epochs::Int
    batch_size::Int
    lr::Float64
    kgmm_kwargs::NamedTuple
    train_max::Int
    langevin_boundary::Float64
    langevin_dt::Float64
    langevin_steps::Int
    langevin_resolution::Int
    langevin_ens::Int
    langevin_burnin::Int
    plot_window::Int
    kde_max_samples::Int
    rel_entropy_points::Int
    normalize_mode::Symbol
    train_copies::Int
    dataset_key::Union{Nothing,String}
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

function unnormalize_data!(X::AbstractMatrix{T}, means::AbstractVector, stds::AbstractVector) where {T<:Real}
    size(X, 1) == length(means) || error("Means length does not match data rows")
    size(X, 1) == length(stds) || error("Stds length does not match data rows")
    @inbounds for i in 1:size(X, 1)
        μ = T(means[i])
        σ = T(stds[i])
        for j in 1:size(X, 2)
            X[i, j] = X[i, j] * σ + μ
        end
    end
    return X
end

function add_mean_field!(X::AbstractMatrix{T}, mean_field::AbstractVector, indices::AbstractVector{Int}) where {T<:Real}
    size(X, 1) == length(indices) || error("Indices length does not match data rows")
    @inbounds for (row_idx, idx) in enumerate(indices)
        offset = T(mean_field[idx])
        for col in 1:size(X, 2)
            X[row_idx, col] += offset
        end
    end
    return X
end

function collect_for_kde(mat::AbstractMatrix, max_samples::Int)
    total_entries = length(mat)
    if max_samples <= 0 || total_entries <= max_samples
        return vec(copy(mat))
    end
    samples_per_column = size(mat, 1)
    max_columns = max(1, max_samples ÷ samples_per_column)
    col_stride = max(1, cld(size(mat, 2), max_columns))
    selected = mat[:, 1:col_stride:size(mat, 2)]
    return vec(copy(selected))
end

function collect_for_kde(vec_samples::AbstractVector, max_samples::Int)
    total_entries = length(vec_samples)
    if max_samples <= 0 || total_entries <= max_samples
        return collect(vec_samples)
    end
    step = max(1, cld(total_entries, max_samples))
    out_len = cld(total_entries, step)
    buffer = Vector{eltype(vec_samples)}(undef, out_len)
    idx_out = 1
    @inbounds for idx in 1:step:total_entries
        buffer[idx_out] = vec_samples[idx]
        idx_out += 1
    end
    resize!(buffer, idx_out - 1)
    return buffer
end

function estimate_pdf_histogram(data::AbstractVector; nbins::Union{Nothing,Int}=nothing, bandwidth::Union{Nothing,Float64}=nothing)
    n = length(data)
    n == 0 && return PDFEstimate(Float64[], Float64[])

    if nbins === nothing
        nbins = clamp(Int(round(sqrt(n) / 2)), 50, 200)
    end

    data_min, data_max = extrema(data)
    if data_min == data_max
        return PDFEstimate([Float64(data_min)], [Inf])
    end

    bin_edges = range(data_min, data_max; length=nbins + 1)
    bin_width = (data_max - data_min) / nbins
    counts = zeros(Float64, nbins)

    for val in data
        if isfinite(val)
            bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins)
            counts[bin_idx] += 1
        end
    end
    counts ./= (n * bin_width)

    if bandwidth === nothing
        bandwidth = (data_max - data_min) / 30.0
    end

    if bandwidth > 0
        sigma_bins = bandwidth / bin_width
        kernel_radius = ceil(Int, 3 * sigma_bins)
        if kernel_radius > 0
            smoothed = zeros(Float64, nbins)
            for i in 1:nbins
                weight_sum = 0.0
                for j in max(1, i - kernel_radius):min(nbins, i + kernel_radius)
                    dist = (i - j) * bin_width
                    weight = exp(-0.5 * (dist / bandwidth)^2)
                    smoothed[i] += counts[j] * weight
                    weight_sum += weight
                end
                smoothed[i] = weight_sum > 0 ? smoothed[i] / weight_sum : counts[i]
            end
            counts = smoothed
        end
    end

    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins))
    return PDFEstimate(x_centers, counts)
end

function determine_value_range(data::AbstractMatrix;
                               clip_fraction::Float64=0.001,
                               max_samples::Int=1_000_000)
    samples = collect_for_kde(data, max_samples)
    samples64 = Float64.(samples)
    filter!(isfinite, samples64)
    isempty(samples64) && return (-1.0, 1.0)
    α = clamp(clip_fraction, 0.0, 0.5)
    lo = quantile(samples64, α)
    hi = quantile(samples64, 1 - α)
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        lo, hi = extrema(samples64)
    end
    if !isfinite(lo) || !isfinite(hi)
        return (-1.0, 1.0)
    end
    if hi <= lo
        padding = max(abs(lo), 1.0) * 1e-3 + 1e-6
        lo -= padding
        hi += padding
    end
    span = hi - lo
    pad = span == 0 ? max(abs(lo), 1.0) * 0.01 : span * 0.02
    return (lo - pad, hi + pad)
end

function compute_averaged_pdfs(data::AbstractMatrix;
                               value_range::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    n_dims, n_times = size(data)

    total_samples = n_dims * n_times
    nbins_uni = clamp(Int(round(sqrt(total_samples) / 2)), 50, 200)

    data_min = 0.0
    data_max = 0.0
    if value_range === nothing
        all_values = Float64[]
        sizehint!(all_values, total_samples)
        for i in 1:n_dims
            append!(all_values, Float64.(data[i, :]))
        end
        data_min, data_max = extrema(all_values)
    else
        data_min, data_max = value_range
    end
    if !isfinite(data_min) || !isfinite(data_max) || data_max <= data_min
        finite_vals = Float64[]
        for i in 1:n_dims
            vals = Float64.(data[i, :])
            for v in vals
                isfinite(v) && push!(finite_vals, v)
            end
        end
        if isempty(finite_vals)
            data_min, data_max = -1.0, 1.0
        else
            data_min, data_max = extrema(finite_vals)
            if data_max <= data_min
                spread = max(abs(data_min), 1.0)
                data_min -= spread
                data_max += spread
            end
        end
    end

    bin_edges = range(data_min, data_max; length=nbins_uni + 1)
    bin_width = (data_max - data_min) / nbins_uni
    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins_uni))

    avg_density = zeros(Float64, nbins_uni)
    for i in 1:n_dims
        samples = vec(Float64.(data[i, :]))
        counts = zeros(Float64, nbins_uni)
        valid_len = 0
        for val in samples
            if isfinite(val) && data_min <= val <= data_max
                bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins_uni)
                counts[bin_idx] += 1
                valid_len += 1
            end
        end
        if valid_len > 0
            counts ./= (valid_len * bin_width)
            avg_density .+= counts
        end
    end
    avg_density ./= n_dims
    avg_univariate = PDFEstimate(x_centers, avg_density)

    distances = [1, 2, 3]
    avg_bivariates = BivariatePDFEstimate[]

    nbins_biv = clamp(Int(round(sqrt(n_times) / 4)), 30, 100)
    x_biv_edges = range(data_min, data_max; length=nbins_biv + 1)
    y_biv_edges = x_biv_edges
    x_biv_width = (data_max - data_min) / nbins_biv
    y_biv_width = x_biv_width
    x_biv_centers = collect(range(data_min + x_biv_width/2, data_max - x_biv_width/2; length=nbins_biv))
    y_biv_centers = x_biv_centers

    for dist in distances
        avg_biv_density = zeros(Float64, nbins_biv, nbins_biv)

        for i in 1:n_dims
            j = mod1(i + dist, n_dims)
            samples_x = vec(Float64.(data[i, :]))
            samples_y = vec(Float64.(data[j, :]))

            density = zeros(Float64, nbins_biv, nbins_biv)
            valid_count = 0

            for t in 1:n_times
                xv = samples_x[t]
                yv = samples_y[t]
                if isfinite(xv) && isfinite(yv) &&
                   data_min <= xv <= data_max &&
                   data_min <= yv <= data_max
                    xi = clamp(searchsortedlast(x_biv_edges, xv), 1, nbins_biv)
                    yi = clamp(searchsortedlast(y_biv_edges, yv), 1, nbins_biv)
                    density[yi, xi] += 1
                    valid_count += 1
                end
            end

            area = x_biv_width * y_biv_width
            if valid_count > 0 && area > 0
                density ./= (valid_count * area)
            end

            avg_biv_density .+= density
        end

        avg_biv_density ./= n_dims
        push!(avg_bivariates, BivariatePDFEstimate(x_biv_centers, y_biv_centers, avg_biv_density))
    end

    return avg_univariate, avg_bivariates
end

function build_plot(series_emp::AbstractVector, series_gen::AbstractVector,
                    X::AbstractMatrix, X_gen::AbstractMatrix,
                    kde_emp::PDFEstimate,
                    kde_gen::PDFEstimate,
                    avg_uni_emp::PDFEstimate,
                    avg_uni_gen::PDFEstimate,
                    avg_biv_emp::Vector{BivariatePDFEstimate},
                    avg_biv_gen::Vector{BivariatePDFEstimate},
                    rel_ent::Real, cfg::KSConfig, cluster_count::Integer,
                    delta_t, observable_idx::Int;
                    save_path::Union{Nothing,String}=nothing)
    save_path === nothing && error("build_plot requires a save_path")
    mkpath(dirname(save_path))
    fig_path = save_path
    mode_count = size(X, 1)
    clusters_tag = max(cluster_count, 0)
    stride_tag = cfg.n_modes

    fig = Figure(size=(1800, 3600), fontsize=28)

    uni_xlim = (minimum(avg_uni_emp.x), maximum(avg_uni_emp.x))
    uni_ylim = (0.0, maximum(avg_uni_emp.density) * 1.1)

    biv_xlim = (minimum(avg_biv_emp[1].x), maximum(avg_biv_emp[1].x))
    biv_ylim = (minimum(avg_biv_emp[1].y), maximum(avg_biv_emp[1].y))

    biv_density_max = 0.0
    for biv_est in vcat(avg_biv_emp, avg_biv_gen)
        biv_density_max = max(biv_density_max, maximum(biv_est.density))
    end
    biv_density_max = max(biv_density_max, eps(Float64))

    time_ax = Axis(fig[1, 1];
        xlabel="t",
        ylabel="u(x*, t)",
        title=@sprintf("Time series (idx=%d)", observable_idx),
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28)
    tshow = minimum((cfg.plot_window, length(series_emp), length(series_gen)))
    dt_plot = isnothing(delta_t) ? 1.0 : delta_t * cfg.stride
    time_vector = (0:(tshow - 1)) .* dt_plot
    lines!(time_ax, time_vector, series_emp[1:tshow]; color=:steelblue, linewidth=2.0, label="Observed")
    lines!(time_ax, time_vector, series_gen[1:tshow]; color=:tomato, linewidth=2.0, linestyle=:dash, label="Generated")
    axislegend(time_ax; position=:rt, framevisible=true, backgroundcolor=:white, labelsize=24)

    univariate_ax = Axis(fig[1, 2];
        xlabel="u",
        ylabel="PDF",
        title="Averaged Univariate PDF",
        titlesize=32,
        xlabelsize=28,
        ylabelsize=28)
    xlims!(univariate_ax, uni_xlim)
    ylims!(univariate_ax, uni_ylim)
    lines!(univariate_ax, avg_uni_emp.x, avg_uni_emp.density; color=:steelblue, linewidth=2.5, label="Observed")
    lines!(univariate_ax, avg_uni_gen.x, avg_uni_gen.density; color=:tomato, linewidth=2.5, linestyle=:dash, label="Generated")
    axislegend(univariate_ax; position=:rt, framevisible=true, backgroundcolor=:white, labelsize=24)

    distances = [1, 2, 3]
    for (row_idx, dist) in enumerate(distances)
        biv_row = row_idx + 1

        ax_obs = Axis(fig[biv_row, 1];
            xlabel="u(x[i])",
            ylabel=@sprintf("u(x[i+%d])", dist),
            title=@sprintf("Observed: <P(x[i],x[i+%d])>ᵢ", dist),
            titlesize=32,
            xlabelsize=28,
            ylabelsize=28)
        xlims!(ax_obs, biv_xlim)
        ylims!(ax_obs, biv_ylim)

        biv_obs = avg_biv_emp[row_idx]
        heatmap!(ax_obs, biv_obs.x, biv_obs.y, biv_obs.density;
                 colormap=:viridis, colorrange=(0, biv_density_max))

        ax_gen = Axis(fig[biv_row, 2];
            xlabel="u(x[i])",
            ylabel=@sprintf("u(x[i+%d])", dist),
            title=@sprintf("Generated: <P(x[i],x[i+%d])>ᵢ", dist),
            titlesize=32,
            xlabelsize=28,
            ylabelsize=28)
        xlims!(ax_gen, biv_xlim)
        ylims!(ax_gen, biv_ylim)

        biv_gen = avg_biv_gen[row_idx]
        heatmap!(ax_gen, biv_gen.x, biv_gen.y, biv_gen.density;
                 colormap=:viridis, colorrange=(0, biv_density_max))
    end

    Colorbar(fig[2:4, 3]; limits=(0, biv_density_max), colormap=:viridis,
             label="Density", ticklabelsize=24, labelsize=26, width=25)

    subtitle = @sprintf("KL = %.4f | modes=%d | stride=%d | clusters=%d",
                        rel_ent, mode_count, stride_tag, clusters_tag)
    Label(fig[0, :], text=subtitle, fontsize=30, font=:bold)

    CairoMakie.save(fig_path, fig)
    @info "Saved diagnostics figure" path=fig_path

    return fig_path
end

function load_config(path::AbstractString)::KSConfig
    params = TOML.parsefile(path)
    kgmm_section = get(params, "kgmm_kwargs", Dict{String,Any}())
    kgmm_kwargs = (
        prob = Float64(get(kgmm_section, "prob", 5e-7)),
        conv_param = Float64(get(kgmm_section, "conv_param", 5e-3)),
        i_max = Int(get(kgmm_section, "i_max", 120)),
        show_progress = Bool(get(kgmm_section, "show_progress", false)),
    )

    dataset_key_val = get(params, "dataset_key", nothing)
    dataset_key = dataset_key_val === nothing || dataset_key_val == "" ? nothing : String(dataset_key_val)

    neurons_vec = Vector{Int}(get(params, "neurons", Int[]))

    KSConfig(
        Int(params["stride"]),
        Int(params["max_snapshots"]),
        Int(params["n_modes"]),
        Int(params["seed"]),
        Float64(params["sigma"]),
        neurons_vec,
        Int(params["n_epochs"]),
        Int(params["batch_size"]),
        Float64(params["lr"]),
        kgmm_kwargs,
        Int(params["train_max"]),
        Float64(params["langevin_boundary"]),
        Float64(params["langevin_dt"]),
        Int(params["langevin_steps"]),
        Int(params["langevin_resolution"]),
        Int(params["langevin_ens"]),
        Int(params["langevin_burnin"]),
        Int(params["plot_window"]),
        Int(params["kde_max_samples"]),
        Int(params["rel_entropy_points"]),
        Symbol(String(params["normalize_mode"])),
        Int(params["train_copies"]),
        dataset_key,
    )
end

function read_run_data(run_dir::AbstractString)
    h5_path = joinpath(run_dir, "output.h5")
    isfile(h5_path) || error("output.h5 not found in $(run_dir)")

    h5open(h5_path, "r") do file
        Xn = read(file, "Xn")
        X_gen_n = read(file, "X_gen_n")
        means = read(file, "reduced_means")
        stds = read(file, "reduced_stds")
        mean_field = read(file, "mean_field")
        indices = collect(Int, read(file, "selected_indices"))
        series_emp = Vector{Float64}(read(file, "series_emp"))
        series_gen = Vector{Float64}(read(file, "series_gen"))
        rel_ent = haskey(file, "relative_entropy") ? read(file, "relative_entropy") : NaN
        cluster_count = haskey(file, "kgmm_cluster_count") ? Int(read(file, "kgmm_cluster_count")) : 0
        delta_t = if haskey(file, "delta_t")
            dt_val = read(file, "delta_t")
            isempty(dt_val) ? nothing : dt_val[1]
        else
            nothing
        end
        kde_emp_x = haskey(file, "kde_emp_x") ? Vector{Float64}(read(file, "kde_emp_x")) : Float64[]
        kde_emp_density = haskey(file, "kde_emp_density") ? Vector{Float64}(read(file, "kde_emp_density")) : Float64[]
        kde_gen_x = haskey(file, "kde_gen_x") ? Vector{Float64}(read(file, "kde_gen_x")) : Float64[]
        kde_gen_density = haskey(file, "kde_gen_density") ? Vector{Float64}(read(file, "kde_gen_density")) : Float64[]

        return (
            Xn=Xn,
            X_gen_n=X_gen_n,
            means=means,
            stds=stds,
            mean_field=mean_field,
            indices=indices,
            series_emp=series_emp,
            series_gen=series_gen,
            rel_ent=rel_ent,
            cluster_count=cluster_count,
            delta_t=delta_t,
            kde_emp_x=kde_emp_x,
            kde_emp_density=kde_emp_density,
            kde_gen_x=kde_gen_x,
            kde_gen_density=kde_gen_density,
        )
    end
end

function parse_run_id(arg::AbstractString)
    cleaned = strip(arg)
    if startswith(cleaned, "run_")
        return parse(Int, cleaned[5:end])
    end
    return parse(Int, cleaned)
end

function main()
    isempty(ARGS) && error("Usage: julia ks_plot.jl <run_number>")
    run_number = parse_run_id(ARGS[1])
    run_name = @sprintf("run_%03d", run_number)
    run_dir = joinpath(@__DIR__, "runs", run_name)
    isdir(run_dir) || error("Run directory $(run_name) not found at $(run_dir)")

    params_path = joinpath(run_dir, "parameters.toml")
    isfile(params_path) || error("parameters.toml not found in $(run_dir)")
    cfg = load_config(params_path)

    data = read_run_data(run_dir)

    Xn = data.Xn
    X_gen_n = data.X_gen_n

    means = data.means
    stds = data.stds
    mean_field = data.mean_field
    indices = data.indices

    unnormalize_data!(Xn, means, stds)
    unnormalize_data!(X_gen_n, means, stds)

    add_mean_field!(Xn, mean_field, indices)
    add_mean_field!(X_gen_n, mean_field, indices)

    empirical_subset = Xn
    generated_subset = X_gen_n

    pdf_emp_samples = collect_for_kde(empirical_subset, cfg.kde_max_samples)
    pdf_gen_samples = collect_for_kde(generated_subset, cfg.kde_max_samples)

    kde_emp = if !isempty(data.kde_emp_x) && !isempty(data.kde_emp_density)
        PDFEstimate(data.kde_emp_x, data.kde_emp_density)
    else
        estimate_pdf_histogram(pdf_emp_samples)
    end
    kde_gen = if !isempty(data.kde_gen_x) && !isempty(data.kde_gen_density)
        PDFEstimate(data.kde_gen_x, data.kde_gen_density)
    else
        estimate_pdf_histogram(pdf_gen_samples)
    end

    range_emp = determine_value_range(empirical_subset)
    range_gen = determine_value_range(generated_subset)
    value_lo = min(range_emp[1], range_gen[1])
    value_hi = max(range_emp[2], range_gen[2])
    if value_hi <= value_lo
        span = max(abs(value_lo), 1.0)
        value_lo -= span
        value_hi += span
    end
    value_range = (value_lo, value_hi)

    avg_uni_emp, avg_biv_emp = compute_averaged_pdfs(empirical_subset; value_range=value_range)
    avg_uni_gen, avg_biv_gen = compute_averaged_pdfs(generated_subset; value_range=value_range)

    series_emp = data.series_emp
    series_gen = data.series_gen
    observable_idx = indices[1]

    figure_path = joinpath(run_dir, "comparison.png")
    build_plot(series_emp, series_gen, empirical_subset, generated_subset,
               kde_emp, kde_gen, avg_uni_emp, avg_uni_gen,
               avg_biv_emp, avg_biv_gen,
               data.rel_ent, cfg, data.cluster_count,
               data.delta_t, observable_idx;
               save_path=figure_path)

    @info "Comparison figure updated" run=run_name path=figure_path
end

isinteractive() || main()
