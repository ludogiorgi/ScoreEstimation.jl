# src/utils.jl
"""
Utility functions for time series analysis, including decorrelation analysis
and data augmentation for circular-symmetric systems.
"""

using Statistics
using FFTW
using Base.Threads

"""
    decorrelation_analysis(timeseries::AbstractMatrix;
                          circular_invariant::Bool=false)
        -> Union{Tuple{Vector{Float64}, Float64}, Tuple{Float64, Float64, Float64}}

Compute decorrelation times for a time series and estimate the number of
uncorrelated samples.

# Arguments
- `timeseries`: DxN matrix where D is the number of dimensions and N is the
                number of time steps
- `circular_invariant`: If true, assumes the system is circular translationally
                       invariant, meaning all dimensions have the same statistics

# Returns
When `circular_invariant=false`:
- `(decorr_times, n_uncorrelated)`: Vector of decorrelation times for each
   dimension and the total number of uncorrelated samples (N divided by the
   largest decorrelation time)

When `circular_invariant=true`:
- `(avg_decorr_time, n_uncorrelated, decorr_scale)`: Average decorrelation time
   across all dimensions, number of uncorrelated samples, and the decorrelation
   scale across dimensions (minimum j such that ⟨x[i]x[i+j]⟩ᵢ ≈ 0)

# Algorithm
Uses FFT-based autocorrelation for each dimension and integrates the normalized
autocorrelation until it becomes non-positive. The decorrelation time τ is
computed as:
    τ = 1 + 2∑ᵏ₌₁ ρ(k)
where ρ(k) is the normalized autocorrelation at lag k (stops when ρ(k) ≤ 0).

For circular invariant systems, also computes the spatial decorrelation scale
by finding the minimum lag j where the average cross-correlation ⟨x[i]x[i+j]⟩ᵢ
becomes non-positive.

# Performance
- Uses FFTW for O(N log N) autocorrelation computation
- Multithreaded processing across dimensions when available
- Preallocated buffers minimize allocations
- Optimized for large time series (millions of samples)

# Example
```julia
# Non-circular system: get per-dimension decorrelation times
data = randn(10, 100000)
decorr_times, n_uncorr = decorrelation_analysis(data; circular_invariant=false)

# Circular-symmetric system: get averaged metrics + spatial scale
data_circular = randn(36, 100000)
avg_tau, n_uncorr, spatial_scale = decorrelation_analysis(data_circular; circular_invariant=true)
```
"""
function decorrelation_analysis(timeseries::AbstractMatrix;
                                circular_invariant::Bool=false)
    n_dim, n_time = size(timeseries)

    if n_dim == 0
        return circular_invariant ? (0.0, 0.0, 0.0) : (Float64[], 0.0)
    end

    if n_time < 2
        if circular_invariant
            return (1.0, 1.0, 1.0)
        else
            return (ones(Float64, n_dim), 1.0)
        end
    end

    # Compute decorrelation times for all dimensions using FFT-based autocorrelation
    decorr_times = _compute_decorrelation_times(timeseries)
    max_decorr_time = maximum(decorr_times)
    n_uncorrelated = n_time / max(max_decorr_time, 1.0)

    if !circular_invariant
        return (decorr_times, n_uncorrelated)
    end

    # For circular invariant systems, compute additional metrics
    avg_decorr_time = mean(decorr_times)
    decorr_scale = _compute_spatial_decorrelation_scale(timeseries)

    return (avg_decorr_time, n_uncorrelated, decorr_scale)
end

"""
    _compute_decorrelation_times(timeseries::AbstractMatrix) -> Vector{Float64}

Internal function to compute decorrelation times for each dimension using
FFT-based autocorrelation. Highly optimized with multithreading and minimal allocations.
"""
function _compute_decorrelation_times(timeseries::AbstractMatrix)
    n_dim, n_time = size(timeseries)

    # Use power-of-2 padding for optimal FFT performance
    padded_len = 1 << ceil(Int, log2(2 * n_time))

    # Precompute normalization factors for autocorrelation
    lag_norm = collect(n_time:-1:1)

    decorr_times = Vector{Float64}(undef, n_dim)

    # Process dimensions in parallel using multithreading
    @threads for dim in 1:n_dim
        # Compute mean and center the data
        vals = @view timeseries[dim, :]
        μ = mean(vals)

        # Prepare padded buffer for FFT
        tmp = zeros(Float64, padded_len)
        @inbounds for t in 1:n_time
            tmp[t] = vals[t] - μ
        end

        # FFT-based autocorrelation:
        # 1. Forward FFT
        freq_buffer = rfft(tmp)
        freq_len = length(freq_buffer)

        # 2. Compute power spectrum (element-wise squared magnitude)
        @inbounds @simd for k in 1:freq_len
            z = freq_buffer[k]
            freq_buffer[k] = z * conj(z)
        end

        # 3. Inverse FFT to get autocorrelation
        ac_buffer = irfft(freq_buffer, padded_len)

        # 4. Normalize by lag count
        @inbounds @simd for lag in 1:n_time
            ac_buffer[lag] /= lag_norm[lag]
        end

        c0 = ac_buffer[1]
        if !(c0 > 0)
            decorr_times[dim] = 1.0
            continue
        end

        # Integrate autocorrelation: τ = 1 + 2∑ρ(k) until ρ(k) ≤ 0
        τ = 1.0
        @inbounds for lag in 2:n_time
            ρ = ac_buffer[lag] / c0
            if ρ <= 0
                break
            end
            τ += 2.0 * ρ
        end
        decorr_times[dim] = τ
    end

    return decorr_times
end

"""
    _compute_spatial_decorrelation_scale(timeseries::AbstractMatrix) -> Float64

Internal function to compute the spatial decorrelation scale for circular
invariant systems. Returns the minimum lag j such that the average
cross-correlation ⟨x[i]x[i+j]⟩ᵢ becomes non-positive.

Uses efficient covariance matrix computation with circular boundary conditions.
"""
function _compute_spatial_decorrelation_scale(timeseries::AbstractMatrix)
    n_dim, n_time = size(timeseries)

    if n_dim <= 1 || n_time == 0
        return 1.0
    end

    # Subsample if time series is very long (for computational efficiency)
    max_samples = 4096
    step = max(1, n_time ÷ max_samples)
    sampled_indices = 1:step:n_time
    sample_count = length(sampled_indices)

    if sample_count <= 1
        return 1.0
    end

    # Compute means for each dimension
    means = zeros(Float64, n_dim)
    @inbounds for idx in sampled_indices
        @simd for i in 1:n_dim
            means[i] += timeseries[i, idx]
        end
    end
    @. means /= sample_count

    # Compute covariance matrix (only upper triangle + diagonal for efficiency)
    cov = zeros(Float64, n_dim, n_dim)
    @inbounds for idx in sampled_indices
        for i in 1:n_dim
            xi = timeseries[i, idx] - means[i]
            # Diagonal
            cov[i, i] += xi * xi
            # Upper triangle
            @simd for j in (i+1):n_dim
                xj = timeseries[j, idx] - means[j]
                cov[i, j] += xi * xj
            end
        end
    end

    # Normalize covariance and mirror to lower triangle
    scale = 1.0 / (sample_count - 1)
    @inbounds for i in 1:n_dim
        cov[i, i] *= scale
        for j in (i+1):n_dim
            cov[i, j] *= scale
            cov[j, i] = cov[i, j]
        end
    end

    # Convert to correlation matrix
    vars = [cov[i, i] for i in 1:n_dim]
    @inbounds for i in 1:n_dim
        if vars[i] <= 0
            vars[i] = eps(Float64)
        end
    end

    @inbounds for i in 1:n_dim
        cov[i, i] = 1.0
        for j in (i+1):n_dim
            denom = sqrt(vars[i] * vars[j])
            val = denom > 0 ? cov[i, j] / denom : 0.0
            val = clamp(val, -1.0, 1.0)
            cov[i, j] = val
            cov[j, i] = val
        end
    end

    # Find decorrelation scale: minimum lag where avg correlation ≤ 0
    # Use circular boundary conditions: x[i] correlates with x[(i+lag) mod n_dim]
    @inbounds for lag in 1:(n_dim - 1)
        total = 0.0
        @simd for i in 1:n_dim
            j = mod1(i + lag, n_dim)
            total += cov[i, j]
        end
        avg_corr = total / n_dim

        if !isfinite(avg_corr) || avg_corr <= 0
            return Float64(lag)
        end
    end

    return Float64(n_dim)
end

"""
    augment_circular_data(timeseries::AbstractMatrix, factor::Int)
        -> Matrix{Float64}

Augment a time series from a circular-symmetric system by using circshift
to create additional samples. Optimally selects dimensions to maximize
decorrelation.

# Arguments
- `timeseries`: DxN matrix where D is the number of dimensions (must be
                circular-symmetric) and N is the number of time steps
- `factor`: Augmentation factor F. Must be a divisor of D (i.e., D/F = k must
           be an integer). The output will have shape Dx(N*F).

# Returns
- Augmented matrix of shape Dx(N*F)

# Algorithm
Instead of simply taking the first F dimensions and applying circshift,
this function optimally selects dimensions to maximize decorrelation:
- If F < D, selects one dimension every k = D/F
- For example, if D=36 and F=9, selects dimensions spaced by k=4
- Each shift applies circshift to the entire spatial dimension, creating
  a translated version of the field that respects circular symmetry

This approach leverages the spatial decorrelation structure of circular-symmetric
systems to generate more independent samples.

# Performance
- O(D*N*F) time complexity
- Minimal allocations (single output array)
- Uses @inbounds and @simd for optimal performance
- Thread-safe for parallel calls

# Example
```julia
# Circular-symmetric system with 36 dimensions, 1000 time steps
data = randn(36, 1000)

# Augment by factor of 9 (36/9 = 4, so one shift every 4 spatial indices)
augmented = augment_circular_data(data, 9)
size(augmented)  # (36, 9000)
```
"""
function augment_circular_data(timeseries::AbstractMatrix, factor::Int)
    n_dim, n_time = size(timeseries)

    # Validate inputs
    if factor <= 0
        throw(ArgumentError("Augmentation factor must be positive, got $factor"))
    end

    if factor == 1
        return copy(timeseries)
    end

    if factor > n_dim
        throw(ArgumentError(
            "Augmentation factor ($factor) cannot exceed number of dimensions ($n_dim)"
        ))
    end

    # Enforce that D/F is an integer
    k = n_dim ÷ factor
    if n_dim != k * factor
        throw(ArgumentError(
            "Number of dimensions ($n_dim) must be divisible by factor ($factor). " *
            "Got D/F = $n_dim/$factor = $(n_dim/factor), which is not an integer."
        ))
    end

    # Allocate output array
    out = Matrix{Float64}(undef, n_dim, n_time * factor)

    # For circular systems, we apply circshift with different shift amounts
    # spaced by k = D/F to maximize decorrelation
    @inbounds for shift_idx in 0:(factor - 1)
        # Column range in output matrix for this shift
        col_start = shift_idx * n_time + 1

        # Shift amount: use spacing of k to maximize spatial decorrelation
        shift_amount = shift_idx * k

        # Apply circshift: circular translation of spatial dimension
        for j in 1:n_time
            @simd for i in 1:n_dim
                # Circular shift: row i reads from row mod1(i + shift_amount, n_dim)
                source_row = mod1(i + shift_amount, n_dim)
                out[i, col_start + j - 1] = timeseries[source_row, j]
            end
        end
    end

    return out
end

"""
    stack_circshifts(data::AbstractMatrix, copies::Int) -> Matrix{Float64}

Legacy function for backward compatibility. Stack circshifts of data
side-by-side along columns.

# Arguments
- `data`: DxN matrix
- `copies`: Number of copies to create (must be ≥ 1)

# Returns
- Dx(N*copies) matrix created by stacking circshifted versions

# Note
This is the simpler version that shifts all dimensions equally. For better
decorrelation in circular-symmetric systems, use `augment_circular_data` instead.

# Example
```julia
data = randn(10, 100)
augmented = stack_circshifts(data, 5)
size(augmented)  # (10, 500)
```
"""
function stack_circshifts(data::AbstractMatrix, copies::Int)
    copies = max(copies, 1)
    copies == 1 && return copy(data)

    nrows, ncols = size(data)
    out = similar(data, nrows, ncols * copies)

    @inbounds for i in 0:(copies - 1)
        # Shift down by i rows (wrap-around), no column shift
        col_start = i * ncols + 1
        col_end = (i + 1) * ncols
        out[:, col_start:col_end] = circshift(data, (i, 0))
    end

    return out
end

"""
    average_decorrelation_length(series::AbstractMatrix) -> Float64

Compute the average decorrelation time across all dimensions of a time series.
This is a convenience wrapper for backward compatibility with existing code.

# Arguments
- `series`: DxN matrix where D is dimensions and N is time steps

# Returns
- Average decorrelation time (scalar)

# Note
For more detailed analysis, use `decorrelation_analysis` directly, which
provides per-dimension decorrelation times and additional metrics.

# Example
```julia
data = randn(10, 10000)
avg_tau = average_decorrelation_length(data)
```
"""
function average_decorrelation_length(series::AbstractMatrix)
    n_dim, n_time = size(series)

    if n_dim == 0
        return 0.0
    end

    if n_time < 2
        return 1.0
    end

    decorr_times = _compute_decorrelation_times(series)
    return mean(decorr_times)
end

# ============================================================================
# PDF Estimation Functions
# ============================================================================

"""
    PDFEstimate

Simple structure to hold univariate PDF estimation results.
Compatible with plotting code.
"""
struct PDFEstimate
    x::Vector{Float64}
    density::Vector{Float64}
end

"""
    BivariatePDFEstimate

Structure to hold bivariate PDF estimation results.
"""
struct BivariatePDFEstimate
    x::Vector{Float64}
    y::Vector{Float64}
    density::Matrix{Float64}
end

"""
    collect_for_kde(mat::AbstractMatrix, max_samples::Int) -> Vector

Subsample a matrix for KDE estimation by taking a subset of columns.
Useful for reducing computational cost when estimating PDFs from large datasets.

# Arguments
- `mat`: DxN matrix to subsample
- `max_samples`: Maximum number of samples to collect (0 = use all)

# Returns
- Flattened vector of selected samples

# Example
```julia
data = randn(10, 100000)
samples = collect_for_kde(data, 10000)  # Subsample to 10k samples
```
"""
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

"""
    collect_for_kde(vec_samples::AbstractVector, max_samples::Int) -> Vector

Subsample a vector for KDE estimation.

# Arguments
- `vec_samples`: Vector to subsample
- `max_samples`: Maximum number of samples to collect (0 = use all)

# Returns
- Vector of selected samples
"""
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

"""
    decorrelation_metrics(series::AbstractVector) -> (Float64, Float64)

Compute decorrelation time and effective sample size for a 1D time series.

# Arguments
- `series`: 1D time series

# Returns
- `(tau, n_effective)`: Decorrelation time and effective number of independent samples

# Example
```julia
timeseries = randn(10000)
tau, n_eff = decorrelation_metrics(timeseries)
println("Decorrelation time: \$tau, Effective samples: \$n_eff")
```
"""
function decorrelation_metrics(series::AbstractVector)
    nsamples = length(series)
    if nsamples == 0
        return (0.0, 0.0)
    end
    series_matrix = reshape(series, 1, nsamples)
    tau = average_decorrelation_length(series_matrix)
    tau = tau > 0 ? tau : 1.0
    effective = nsamples / tau
    return (tau, effective)
end

"""
    estimate_pdf_histogram(data; nbins=nothing, bandwidth=nothing) -> PDFEstimate

Estimate a univariate PDF using histogram binning with Gaussian smoothing.
Optimized replacement for KernelDensity.jl's kde() function.

# Arguments
- `data`: Vector of samples
- `nbins`: Number of bins (default: sqrt(n)/2 clamped to 50-200)
- `bandwidth`: Smoothing bandwidth in data units (default: adaptive based on data range)

# Returns
- `PDFEstimate`: Structure containing grid points and density values

# Performance
- O(n + b²) where n=samples, b=bins
- Uses Gaussian kernel smoothing for smooth density estimates
- Memory-efficient histogram-based approach

# Example
```julia
samples = randn(10000)
pdf_est = estimate_pdf_histogram(samples)
# Plot: plot(pdf_est.x, pdf_est.density)
```
"""
function estimate_pdf_histogram(data::AbstractVector; nbins::Union{Nothing,Int}=nothing, bandwidth::Union{Nothing,Float64}=nothing)
    n = length(data)
    n == 0 && return PDFEstimate(Float64[], Float64[])

    # Determine number of bins
    if nbins === nothing
        nbins = clamp(Int(round(sqrt(n) / 2)), 50, 200)
    end

    # Get data range
    data_min, data_max = extrema(data)
    if data_min == data_max
        # Degenerate case: all values are the same
        return PDFEstimate([data_min], [Inf])
    end

    # Create histogram
    bin_edges = range(data_min, data_max; length=nbins + 1)
    bin_width = (data_max - data_min) / nbins
    counts = zeros(Float64, nbins)

    # Fill histogram
    @inbounds for val in data
        if isfinite(val)
            bin_idx = clamp(searchsortedlast(bin_edges, val), 1, nbins)
            counts[bin_idx] += 1
        end
    end

    # Normalize to get density
    counts ./= (n * bin_width)

    # Apply Gaussian smoothing if requested
    if bandwidth === nothing
        # Adaptive bandwidth: use a fraction of the data range
        bandwidth = (data_max - data_min) / 30.0
    end

    if bandwidth > 0
        # Smooth the histogram with a Gaussian kernel
        sigma_bins = bandwidth / bin_width  # Convert to bin units
        kernel_radius = ceil(Int, 3 * sigma_bins)  # 3-sigma cutoff

        if kernel_radius > 0
            smoothed = zeros(Float64, nbins)
            @inbounds for i in 1:nbins
                weight_sum = 0.0
                for j in max(1, i - kernel_radius):min(nbins, i + kernel_radius)
                    dist = (i - j) * bin_width
                    weight = exp(-0.5 * (dist / bandwidth)^2)
                    smoothed[i] += counts[j] * weight
                    weight_sum += weight
                end
                smoothed[i] /= weight_sum
            end
            counts = smoothed
        end
    end

    # Create grid points at bin centers
    x_centers = collect(range(data_min + bin_width/2, data_max - bin_width/2; length=nbins))

    return PDFEstimate(x_centers, counts)
end

"""
    estimate_bivariate_pdf_histogram(data_x, data_y; nbins=nothing, x_range=nothing, y_range=nothing) -> BivariatePDFEstimate

Estimate a bivariate PDF using 2D histogram binning.

# Arguments
- `data_x`: Vector of x samples
- `data_y`: Vector of y samples
- `nbins`: Number of bins per dimension (default: sqrt(n)/4 clamped to 30-100)
- `x_range`: Tuple (x_min, x_max) to define x-axis range (default: data extrema)
- `y_range`: Tuple (y_min, y_max) to define y-axis range (default: data extrema)

# Returns
- `BivariatePDFEstimate`: Structure containing grid points and 2D density

# Performance
- O(n + b²) where n=samples, b=bins per dimension
- Memory: O(b²) for density matrix

# Example
```julia
x = randn(10000)
y = randn(10000)
pdf_2d = estimate_bivariate_pdf_histogram(x, y)
# Plot: heatmap(pdf_2d.x, pdf_2d.y, pdf_2d.density)
```
"""
function estimate_bivariate_pdf_histogram(data_x::AbstractVector, data_y::AbstractVector;
                                         nbins::Union{Nothing,Int}=nothing,
                                         x_range::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                                         y_range::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    n = length(data_x)
    @assert length(data_y) == n "data_x and data_y must have the same length"
    n == 0 && return BivariatePDFEstimate(Float64[], Float64[], zeros(Float64, 0, 0))

    # Determine number of bins
    if nbins === nothing
        nbins = clamp(Int(round(sqrt(n) / 4)), 30, 100)
    end

    # Get data ranges
    if x_range === nothing
        x_min, x_max = extrema(data_x)
    else
        x_min, x_max = x_range
    end

    if y_range === nothing
        y_min, y_max = extrema(data_y)
    else
        y_min, y_max = y_range
    end

    if x_min == x_max || y_min == y_max
        return BivariatePDFEstimate([x_min], [y_min], zeros(Float64, 1, 1))
    end

    # Create histogram
    x_edges = range(x_min, x_max; length=nbins + 1)
    y_edges = range(y_min, y_max; length=nbins + 1)
    x_width = (x_max - x_min) / nbins
    y_width = (y_max - y_min) / nbins
    density = zeros(Float64, nbins, nbins)

    # Fill histogram
    valid_count = 0
    @inbounds for i in 1:n
        xv = data_x[i]
        yv = data_y[i]
        if isfinite(xv) && isfinite(yv) && x_min <= xv <= x_max && y_min <= yv <= y_max
            xi = clamp(searchsortedlast(x_edges, xv), 1, nbins)
            yi = clamp(searchsortedlast(y_edges, yv), 1, nbins)
            density[yi, xi] += 1
            valid_count += 1
        end
    end

    # Normalize to get density
    area = x_width * y_width
    if valid_count > 0 && area > 0
        density ./= (valid_count * area)
    end

    # Create grid points at bin centers
    x_centers = collect(range(x_min + x_width/2, x_max - x_width/2; length=nbins))
    y_centers = collect(range(y_min + y_width/2, y_max - y_width/2; length=nbins))

    return BivariatePDFEstimate(x_centers, y_centers, density)
end

"""
    determine_value_range(data; clip_fraction=0.001, max_samples=1_000_000)
        -> Tuple{Float64, Float64}

Estimate a robust value range for histogramming by clipping extreme outliers.
Returns a tuple `(lo, hi)` expanded by a small safety margin.

# Arguments
- `data`: Data matrix
- `clip_fraction`: Fraction of extreme values to clip (default: 0.001 = 0.1%)
- `max_samples`: Maximum samples to use for quantile estimation

# Returns
- `(lo, hi)`: Robust value range with padding

# Example
```julia
data = randn(100, 1000)
lo, hi = determine_value_range(data)
# Use for consistent histogram ranges across datasets
```
"""
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

"""
    compute_averaged_pdfs(data::AbstractMatrix; value_range=nothing)
        -> (PDFEstimate, Vector{BivariatePDFEstimate})

Compute averaged univariate and bivariate PDFs using circular translational
invariance. Designed for systems with periodic boundary conditions.

# Arguments
- `data`: DxN matrix where D is spatial dimensions (periodic), N is time steps
- `value_range`: Optional `(lo, hi)` tuple to enforce consistent histogram range

# Returns
- `(avg_univariate, avg_bivariates)`: Averaged PDF and vector of bivariate PDFs
  for spatial distances [1, 2, 3]

# Algorithm
Leverages circular symmetry to average over all spatial dimensions:
- Univariate: averages marginal PDF across all D dimensions
- Bivariate: averages joint PDF P(x[i], x[i+d]) for d ∈ {1,2,3} using mod1

# Performance
- O(D*N*b) for univariate, O(D*N*b²) for bivariate where b=bins
- Optimized with @inbounds for tight loops

# Example
```julia
# Periodic 1D field (e.g., Kuramoto-Sivashinsky)
field = randn(32, 10000)  # 32 spatial points, 10k timesteps
avg_uni, avg_biv = compute_averaged_pdfs(field)

# avg_biv[1] = P(x[i], x[i+1]) averaged over i
# avg_biv[2] = P(x[i], x[i+2]) averaged over i
# avg_biv[3] = P(x[i], x[i+3]) averaged over i
```
"""
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
        @inbounds for i in 1:n_dims
            append!(all_values, Float64.(data[i, :]))
        end
        data_min, data_max = extrema(all_values)
    else
        data_min, data_max = value_range
    end
    if !isfinite(data_min) || !isfinite(data_max) || data_max <= data_min
        finite_vals = Float64[]
        @inbounds for i in 1:n_dims
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
    @inbounds for i in 1:n_dims
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

    @inbounds for dist in distances
        avg_biv_density = zeros(Float64, nbins_biv, nbins_biv)

        for i in 1:n_dims
            j = mod1(i + dist, n_dims)  # Circular indexing
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
