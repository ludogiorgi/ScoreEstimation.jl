Base.@kwdef struct ResidualResult
    times::Vector{Float64}
    x_smoothed::Matrix{Float64}
    dx_estimated::Matrix{Float64}
    drift_estimated::Matrix{Float64}
    residual::Matrix{Float64}
    true_residual::Matrix{Float64}
    metrics_r1
    metrics_r2
end

function local_polyfit_smooth_and_derivative(times::AbstractVector{<:Real},
                                             values::AbstractVector{<:Real};
                                             window_radius::Integer=3,
                                             degree::Integer=2)
    n = length(times)
    n == length(values) || error("times and values must have the same length.")
    degree_eff = min(Int(degree), max(n - 1, 0))
    min_points = max(degree_eff + 1, 2)
    smoothed = Vector{Float64}(undef, n)
    derivative = Vector{Float64}(undef, n)
    t = Float64.(times)
    y = Float64.(values)
    for idx in 1:n
        lo = max(1, idx - window_radius)
        hi = min(n, idx + window_radius)
        while (hi - lo + 1) < min_points
            if lo > 1
                lo -= 1
            elseif hi < n
                hi += 1
            else
                break
            end
        end
        inds = lo:hi
        τ = t[inds] .- t[idx]
        design = hcat([τ .^ p for p in 0:degree_eff]...)
        coeffs = design \ y[inds]
        smoothed[idx] = coeffs[1]
        derivative[idx] = degree_eff >= 1 ? coeffs[2] : 0.0
    end
    return smoothed, derivative
end

function smooth_state_and_derivative(data::SimulationData, cfg::ResidualConfig)
    x1_smooth, dx1 = local_polyfit_smooth_and_derivative(data.t, data.x1;
        window_radius=cfg.derivative_window_radius,
        degree=cfg.derivative_degree,
    )
    x2_smooth, dx2 = local_polyfit_smooth_and_derivative(data.t, data.x2;
        window_radius=cfg.derivative_window_radius,
        degree=cfg.derivative_degree,
    )
    trim = max(cfg.derivative_window_radius, cfg.derivative_degree)
    valid = (1 + trim):(length(data.t) - trim)
    times = data.t[valid]
    x_smoothed = hcat(x1_smooth[valid], x2_smooth[valid])'
    dx_estimated = hcat(dx1[valid], dx2[valid])'
    return times, x_smoothed, dx_estimated, collect(valid)
end

function true_residual_components(data::SimulationData, params::ModelParameters)
    force_x1 = params.forcing_x1 === :y2 ? (data.y2 .- params.y2_ref) : (data.y3 .- params.y3_ref)
    force_x2 = params.forcing_x2 === :y2 ? (data.y2 .- params.y2_ref) : (data.y3 .- params.y3_ref)
    return Matrix{Float64}(vcat(
        permutedims(params.sigma_x1 .* force_x1),
        permutedims(params.sigma_x2 .* force_x2),
    ))
end

function regression_metrics(observed::AbstractVector{<:Real}, estimated::AbstractVector{<:Real})
    design = hcat(ones(length(observed)), Float64.(observed))
    β = design \ Float64.(estimated)
    residual = Float64.(estimated) .- design * β
    rmse = sqrt(mean(abs2, residual))
    return (intercept=β[1], slope=β[2], rmse=rmse)
end

function histogram1d(values::AbstractVector{<:Real}, nbins::Int; limits=nothing)
    lo, hi = limits === nothing ? extrema(values) : limits
    pad = max(0.05 * (hi - lo), 1.0e-6)
    edges = collect(range(lo - pad, hi + pad, length=nbins + 1))
    counts = zeros(Float64, nbins)
    for value in values
        idx = clamp(searchsortedlast(edges, value), 1, nbins)
        counts[idx] += 1
    end
    width = edges[2] - edges[1]
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    density = counts ./ (sum(counts) * width)
    return centers, density
end

function discrete_kl(p::AbstractVector{<:Real}, q::AbstractVector{<:Real}; floor::Float64=1.0e-12)
    p_safe = max.(Float64.(p), floor)
    q_safe = max.(Float64.(q), floor)
    p_safe ./= sum(p_safe)
    q_safe ./= sum(q_safe)
    return sum(p_safe .* log.(p_safe ./ q_safe))
end

function autocorrelation(series::AbstractVector{<:Real}, maxlag::Int)
    n = length(series)
    maxlag = min(maxlag, n - 1)
    centered = Float64.(series) .- mean(series)
    variance = mean(abs2, centered)
    variance > 0 || return zeros(Float64, maxlag + 1)
    out = zeros(Float64, maxlag + 1)
    for lag in 0:maxlag
        out[lag + 1] = dot(view(centered, 1:(n - lag)), view(centered, 1 + lag:n)) / ((n - lag) * variance)
    end
    return out
end

function compute_residual_metrics(true_residual::AbstractVector{<:Real},
                                  estimated_residual::AbstractVector{<:Real},
                                  cfg::ResidualConfig,
                                  dt::Real)
    obs = Float64.(true_residual)
    est = Float64.(estimated_residual)
    reg = regression_metrics(obs, est)
    common_limits = (min(minimum(obs), minimum(est)), max(maximum(obs), maximum(est)))
    grid, pdf_obs = histogram1d(obs, cfg.pdf_bins; limits=common_limits)
    _, pdf_est = histogram1d(est, cfg.pdf_bins; limits=common_limits)
    common_acf_obs = autocorrelation(obs, cfg.acf_lags)
    common_acf_est = autocorrelation(est, cfg.acf_lags)
    lag_axis = collect(0:length(common_acf_obs)-1) .* dt
    return (
        correlation=cor(obs, est),
        slope=reg.slope,
        intercept=reg.intercept,
        rmse=sqrt(mean(abs2, est .- obs)),
        normalized_rmse=sqrt(mean(abs2, est .- obs)) / max(std(obs), eps(Float64)),
        pdf_kl=discrete_kl(pdf_obs, pdf_est),
        acf_rmse=sqrt(mean(abs2, common_acf_obs .- common_acf_est)),
        grid=grid,
        pdf_obs=pdf_obs,
        pdf_est=pdf_est,
        lag_axis=lag_axis,
        acf_obs=common_acf_obs,
        acf_est=common_acf_est,
    )
end

function recover_residuals(data::SimulationData, params::ModelParameters, score_model, phi::AbstractMatrix, cfg::ResidualConfig)
    times, x_smoothed, dx_estimated, valid = smooth_state_and_derivative(data, cfg)
    score_input = Float32.(x_smoothed)
    use_gpu = hasproperty(score_model, :model) &&
        hasproperty(score_model.model, :layers) &&
        !isempty(score_model.model.layers) &&
        hasproperty(first(score_model.model.layers), :weight) &&
        !(first(score_model.model.layers).weight isa Array)
    score_values = Array(score_model(use_gpu ? ScoreEstimation._to_device(score_input; use_gpu=true) : score_input))
    drift_estimated = Matrix{Float64}(phi * score_values)
    residual = dx_estimated .- drift_estimated
    true_residual = true_residual_components(data, params)[:, valid]
    metrics_r1 = compute_residual_metrics(vec(true_residual[1, :]), vec(residual[1, :]), cfg, data.sample_dt)
    metrics_r2 = compute_residual_metrics(vec(true_residual[2, :]), vec(residual[2, :]), cfg, data.sample_dt)
    return ResidualResult(
        times=times,
        x_smoothed=x_smoothed,
        dx_estimated=dx_estimated,
        drift_estimated=drift_estimated,
        residual=residual,
        true_residual=true_residual,
        metrics_r1=metrics_r1,
        metrics_r2=metrics_r2,
    )
end

function write_residual_plot_data(dir::String, result::ResidualResult, cfg::ResidualConfig, metrics_path::String)
    mkpath(dir)
    write_vector_csv(
        joinpath(dir, "residual_traj.csv"),
        result.times,
        vec(result.dx_estimated[1, :]),
        vec(result.dx_estimated[2, :]),
        vec(result.true_residual[1, :]),
        vec(result.true_residual[2, :]),
        vec(result.residual[1, :]),
        vec(result.residual[2, :]),
    )
    write_vector_csv(joinpath(dir, "r1_pdf.csv"), result.metrics_r1.grid, result.metrics_r1.pdf_obs, result.metrics_r1.pdf_est)
    write_vector_csv(joinpath(dir, "r2_pdf.csv"), result.metrics_r2.grid, result.metrics_r2.pdf_obs, result.metrics_r2.pdf_est)
    write_vector_csv(joinpath(dir, "r1_acf.csv"), result.metrics_r1.lag_axis, result.metrics_r1.acf_obs, result.metrics_r1.acf_est)
    write_vector_csv(joinpath(dir, "r2_acf.csv"), result.metrics_r2.lag_axis, result.metrics_r2.acf_obs, result.metrics_r2.acf_est)
    write_metric_text(metrics_path, [
        @sprintf("r1 corr = %.5f", result.metrics_r1.correlation),
        @sprintf("r1 slope = %.5f", result.metrics_r1.slope),
        @sprintf("r1 RMSE = %.5f", result.metrics_r1.rmse),
        @sprintf("r1 nRMSE = %.5f", result.metrics_r1.normalized_rmse),
        @sprintf("r1 KL(pdf) = %.5f", result.metrics_r1.pdf_kl),
        @sprintf("r1 ACF RMSE = %.5f", result.metrics_r1.acf_rmse),
        @sprintf("r2 corr = %.5f", result.metrics_r2.correlation),
        @sprintf("r2 slope = %.5f", result.metrics_r2.slope),
        @sprintf("r2 RMSE = %.5f", result.metrics_r2.rmse),
        @sprintf("r2 nRMSE = %.5f", result.metrics_r2.normalized_rmse),
        @sprintf("r2 KL(pdf) = %.5f", result.metrics_r2.pdf_kl),
        @sprintf("r2 ACF RMSE = %.5f", result.metrics_r2.acf_rmse),
        @sprintf("excerpt points = %d", cfg.excerpt_points),
    ])
    return nothing
end
