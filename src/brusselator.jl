using FFTW
using FastSDE

Base.@kwdef struct BrusselatorConfig
    A::Float64 = 1.0
    B::Float64 = 1.9
    Ω::Float64 = 10.0
    dt::Float64 = 1.0e-3
    burnin_steps::Int = 100_000
    n_steps_timeseries::Int = 400_000
    resolution_timeseries::Int = 20
    n_steps_stationary::Int = 200_000
    burnin_stationary::Int = 40_000
    resolution_stationary::Int = 20
    ensemble_stationary::Int = 64
    maxlag::Int = 5_000
    plot_window::Int = 4_000
    phase_points::Int = 15_000
    seed::Int = 2026
    verbose::Bool = true
end

@inline fixed_point(cfg::BrusselatorConfig) = [cfg.A, cfg.B / cfg.A]

function brusselator_drift!(du, u, cfg::BrusselatorConfig, t)
    x = u[1]
    y = u[2]
    du[1] = cfg.A - (cfg.B + 1.0) * x + x * x * y
    du[2] = cfg.B * x - x * x * y
    return nothing
end

@inline function _sqrt_psd_2x2!(Σ::AbstractMatrix, g11::Float64, g12::Float64, g22::Float64)
    τ = 0.5 * (g11 + g22)
    δ = 0.5 * (g11 - g22)
    radius = hypot(δ, g12)
    λ1 = max(τ + radius, 0.0)
    λ2 = max(τ - radius, 0.0)
    θ = 0.5 * atan(2.0 * g12, g11 - g22)
    c = cos(θ)
    s = sin(θ)
    r1 = sqrt(λ1)
    r2 = sqrt(λ2)

    Σ[1, 1] = c * c * r1 + s * s * r2
    Σ[1, 2] = c * s * (r1 - r2)
    Σ[2, 1] = Σ[1, 2]
    Σ[2, 2] = s * s * r1 + c * c * r2
    return nothing
end

function brusselator_sigma!(Σ::AbstractMatrix, u, cfg::BrusselatorConfig, t)
    x = u[1]
    y = u[2]
    reaction_flux = cfg.B * x + x * x * y
    g11 = (cfg.A + x + reaction_flux) / cfg.Ω
    g12 = -reaction_flux / cfg.Ω
    g22 = reaction_flux / cfg.Ω
    return _sqrt_psd_2x2!(Σ, g11, g12, g22)
end

function brusselator_sigma(u, cfg::BrusselatorConfig, t)
    Σ = Matrix{Float64}(undef, 2, 2)
    brusselator_sigma!(Σ, u, cfg, t)
    return Σ
end

function _trim_burnin(saved::AbstractMatrix, burnin_steps::Int, resolution::Int)
    burnin_steps <= 0 && return copy(saved)
    dropped = fld(burnin_steps, resolution) + 1
    start_idx = min(dropped + 1, size(saved, 2))
    return copy(saved[:, start_idx:end])
end

function integrate_brusselator_timeseries(cfg::BrusselatorConfig; u0=fixed_point(cfg))
    total_steps = cfg.burnin_steps + cfg.n_steps_timeseries
    drift! = (du, u, t) -> brusselator_drift!(du, u, cfg, t)
    sigma! = (Σ, u, t) -> brusselator_sigma!(Σ, u, cfg, t)
    raw = evolve(
        u0,
        cfg.dt,
        total_steps,
        drift!,
        sigma!;
        timestepper=:euler,
        resolution=cfg.resolution_timeseries,
        sigma_inplace=true,
        manage_blas_threads=false,
        verbose=cfg.verbose,
    )
    trimmed = _trim_burnin(raw, cfg.burnin_steps, cfg.resolution_timeseries)
    sample_dt = cfg.dt * cfg.resolution_timeseries
    time_axis = collect(0:(size(trimmed, 2) - 1)) .* sample_dt
    return Array(trimmed), time_axis
end

function integrate_brusselator_stationary(cfg::BrusselatorConfig; u0=fixed_point(cfg), flatten::Bool=true)
    drift! = (du, u, t) -> brusselator_drift!(du, u, cfg, t)
    sigma! = (Σ, u, t) -> brusselator_sigma!(Σ, u, cfg, t)
    samples = evolve(
        u0,
        cfg.dt,
        cfg.n_steps_stationary,
        drift!,
        sigma!;
        timestepper=:euler,
        resolution=cfg.resolution_stationary,
        n_burnin=cfg.burnin_stationary,
        n_ens=cfg.ensemble_stationary,
        sigma_inplace=true,
        manage_blas_threads=true,
        flatten=flatten,
        verbose=cfg.verbose,
    )
    return Array(samples)
end

function estimate_uncorrelated_stride(samples::AbstractMatrix)
    taus = map(i -> decorrelation_metrics(view(samples, i, :))[1], axes(samples, 1))
    stride = max(1, ceil(Int, maximum(taus)))
    return stride, Float64.(taus)
end

function subsample_uncorrelated(samples::AbstractMatrix, stride::Integer)
    step = max(Int(stride), 1)
    return copy(samples[:, 1:step:size(samples, 2)])
end

function _robust_range(samples::AbstractVector; clip_fraction::Float64=0.002)
    vals = Float64.(collect(samples))
    filter!(isfinite, vals)
    isempty(vals) && return (-1.0, 1.0)
    lo = quantile(vals, clip_fraction)
    hi = quantile(vals, 1.0 - clip_fraction)
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        lo, hi = extrema(vals)
    end
    pad = max(0.05 * (hi - lo), 1e-3)
    return (lo - pad, hi + pad)
end

function _autocorrelation_fft(series::AbstractVector, maxlag::Int)
    n = length(series)
    n <= 1 && return zeros(Float64, 0)
    m = min(maxlag, n - 1)
    centered = Float64.(series) .- mean(series)
    padded_len = 1 << ceil(Int, log2(max(2, 2 * n)))
    tmp = zeros(Float64, padded_len)
    @views tmp[1:n] .= centered
    freq = rfft(tmp)
    @. freq = freq * conj(freq)
    ac = irfft(freq, padded_len)[1:(m + 1)]
    norm = collect(n:-1:(n - m))
    ac ./= norm
    ac0 = ac[1]
    ac0 > 0 || return zeros(Float64, m + 1)
    ac ./= ac0
    return ac
end

function _cross_correlation(series_x::AbstractVector, series_y::AbstractVector, maxlag::Int)
    n = min(length(series_x), length(series_y))
    n <= 1 && return Int[], Float64[]
    maxlag = min(maxlag, n - 1)
    x = Float64.(series_x[1:n])
    y = Float64.(series_y[1:n])
    x .-= mean(x)
    y .-= mean(y)
    denom = max(std(x) * std(y), eps(Float64))
    lags = collect(-maxlag:maxlag)
    corr = Vector{Float64}(undef, length(lags))

    @inbounds for (idx, lag) in enumerate(lags)
        if lag >= 0
            upper = n - lag
            corr[idx] = upper > 0 ? dot(view(x, 1 + lag:n), view(y, 1:upper)) / (upper * denom) : 0.0
        else
            shift = -lag
            upper = n - shift
            corr[idx] = upper > 0 ? dot(view(x, 1:upper), view(y, 1 + shift:n)) / (upper * denom) : 0.0
        end
    end

    return lags, corr
end

function _power_spectrum(series::AbstractVector, dt_sample::Float64)
    n = length(series)
    n <= 1 && return Float64[], Float64[]
    centered = Float64.(series) .- mean(series)
    spectrum = abs2.(rfft(centered)) ./ n
    if length(spectrum) > 2
        @views spectrum[2:(end - 1)] .*= 2
    end
    freqs = collect(range(0.0, step=1.0 / (n * dt_sample), length=length(spectrum)))
    return freqs, spectrum
end

function build_brusselator_diagnostics(cfg::BrusselatorConfig,
                                       trajectory::AbstractMatrix,
                                       stationary::AbstractMatrix;
                                       sample_dt::Real=cfg.dt * cfg.resolution_timeseries,
                                       x_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                                       y_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                                       pdf_bins::Int=120,
                                       joint_pdf_bins::Int=80)
    x = vec(trajectory[1, :])
    y = vec(trajectory[2, :])
    xs = vec(stationary[1, :])
    ys = vec(stationary[2, :])

    x_support = x_range === nothing ? _robust_range(xs) : x_range
    y_support = y_range === nothing ? _robust_range(ys) : y_range

    pdf_x = estimate_pdf_histogram(xs; nbins=pdf_bins, x_range=x_support)
    pdf_y = estimate_pdf_histogram(ys; nbins=pdf_bins, x_range=y_support)
    pdf_xy = estimate_bivariate_pdf_histogram(xs, ys; nbins=joint_pdf_bins, x_range=x_support, y_range=y_support)

    acf_x = _autocorrelation_fft(x, cfg.maxlag)
    acf_y = _autocorrelation_fft(y, cfg.maxlag)
    lag_axis = collect(0:(length(acf_x) - 1)) .* sample_dt

    ccf_lags, ccf = _cross_correlation(x, y, cfg.maxlag)
    ccf_axis = Float64.(ccf_lags) .* sample_dt

    freq_x, psd_x = _power_spectrum(x, sample_dt)
    freq_y, psd_y = _power_spectrum(y, sample_dt)

    tau_x, neff_x = decorrelation_metrics(x)
    tau_y, neff_y = decorrelation_metrics(y)

    return (
        pdf_x=pdf_x,
        pdf_y=pdf_y,
        pdf_xy=pdf_xy,
        lag_axis=lag_axis,
        acf_x=acf_x,
        acf_y=acf_y,
        ccf_axis=ccf_axis,
        ccf=ccf,
        freq_x=freq_x,
        psd_x=psd_x,
        freq_y=freq_y,
        psd_y=psd_y,
        x_mean=mean(xs),
        y_mean=mean(ys),
        x_std=std(xs),
        y_std=std(ys),
        corr_xy=cor(xs, ys),
        tau_x=tau_x,
        tau_y=tau_y,
        neff_x=neff_x,
        neff_y=neff_y,
        phase_count=min(cfg.phase_points, size(trajectory, 2)),
        x_range=x_support,
        y_range=y_support,
    )
end

@inline function _pdf_mass(pdf::PDFEstimate)
    length(pdf.x) <= 1 && return copy(pdf.density)
    dx = pdf.x[2] - pdf.x[1]
    mass = max.(pdf.density, 0.0) .* dx
    total = sum(mass)
    total > 0 || return fill(1.0 / length(mass), length(mass))
    return mass ./ total
end

@inline function _pdf_mass(pdf::BivariatePDFEstimate)
    nx = length(pdf.x)
    ny = length(pdf.y)
    (nx <= 1 || ny <= 1) && return copy(pdf.density)
    dx = pdf.x[2] - pdf.x[1]
    dy = pdf.y[2] - pdf.y[1]
    mass = max.(pdf.density, 0.0) .* (dx * dy)
    total = sum(mass)
    total > 0 || return fill(1.0 / length(mass), size(mass))
    return mass ./ total
end

@inline function _discrete_kl(p::AbstractArray, q::AbstractArray; floor::Float64=1.0e-12)
    p_safe = max.(p, floor)
    q_safe = max.(q, floor)
    return sum(p_safe .* log.(p_safe ./ q_safe))
end

@inline function _l1_distance(p::AbstractArray, q::AbstractArray)
    return sum(abs.(p .- q))
end

function compare_brusselator_diagnostics(reference,
                                         candidate;
                                         acf_window::Union{Nothing,Int}=nothing,
                                         ccf_window::Union{Nothing,Int}=nothing,
                                         floor::Float64=1.0e-12)
    ref_pdf_x = _pdf_mass(reference.pdf_x)
    ref_pdf_y = _pdf_mass(reference.pdf_y)
    ref_pdf_xy = _pdf_mass(reference.pdf_xy)
    cand_pdf_x = _pdf_mass(candidate.pdf_x)
    cand_pdf_y = _pdf_mass(candidate.pdf_y)
    cand_pdf_xy = _pdf_mass(candidate.pdf_xy)

    acf_count = acf_window === nothing ? min(length(reference.acf_x), length(candidate.acf_x)) :
        min(acf_window, length(reference.acf_x), length(candidate.acf_x))
    ccf_count = ccf_window === nothing ? min(length(reference.ccf), length(candidate.ccf)) :
        min(ccf_window, length(reference.ccf), length(candidate.ccf))

    acf_x_error = reference.acf_x[1:acf_count] .- candidate.acf_x[1:acf_count]
    acf_y_error = reference.acf_y[1:acf_count] .- candidate.acf_y[1:acf_count]
    ccf_error = reference.ccf[1:ccf_count] .- candidate.ccf[1:ccf_count]
    finite_ccf_error = filter(isfinite, ccf_error)

    return (
        kl_x=_discrete_kl(ref_pdf_x, cand_pdf_x; floor=floor),
        kl_y=_discrete_kl(ref_pdf_y, cand_pdf_y; floor=floor),
        kl_xy=_discrete_kl(ref_pdf_xy, cand_pdf_xy; floor=floor),
        l1_x=_l1_distance(ref_pdf_x, cand_pdf_x),
        l1_y=_l1_distance(ref_pdf_y, cand_pdf_y),
        l1_xy=_l1_distance(ref_pdf_xy, cand_pdf_xy),
        rmse_acf_x=sqrt(mean(abs2, acf_x_error)),
        rmse_acf_y=sqrt(mean(abs2, acf_y_error)),
        rmse_ccf=isempty(finite_ccf_error) ? NaN : sqrt(mean(abs2, finite_ccf_error)),
        short_lag_points=acf_count,
        short_ccf_points=ccf_count,
    )
end
