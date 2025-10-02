"""
    relative_entropy(series1, series2; kwargs...)

Compute the relative entropy (KL divergence) between two time series.

# Arguments
- `series1`: First time series (1D vector or dD matrix with dimensions as rows)
- `series2`: Second time series (1D vector or dD matrix with dimensions as rows)

# Keyword Arguments
- `npoints::Int=2048`: Number of points for PDF estimation grid
- `bandwidth::Union{Nothing,Real}=nothing`: KDE bandwidth (auto if nothing)

# Returns
- Scalar KL divergence if inputs are 1D vectors
- Vector of length d containing KL divergence for each dimension if inputs are dD

# Details
The relative entropy (KL divergence) is computed as:
    D_KL(P || Q) = ∫ p(x) log(p(x)/q(x)) dx

where p is the PDF of series1 and q is the PDF of series2.

# Examples
```julia
# 1D case
x1 = randn(1000)
x2 = randn(1000) .+ 0.5
kl = relative_entropy(x1, x2)

# Multi-dimensional case
X1 = randn(3, 1000)  # 3 dimensions
X2 = randn(3, 1000) .+ 0.5
kl_vec = relative_entropy(X1, X2)  # Returns length-3 vector
```
"""
function relative_entropy(series1, series2; npoints::Int=2048, bandwidth::Union{Nothing,Real}=nothing)
    # Handle 1D case
    if series1 isa AbstractVector && series2 isa AbstractVector
        return _kl_divergence_1d(series1, series2, npoints, bandwidth)
    end

    # Handle multi-dimensional case
    if series1 isa AbstractMatrix && series2 isa AbstractMatrix
        d1, n1 = size(series1)
        d2, n2 = size(series2)

        if d1 != d2
            throw(DimensionMismatch("series1 has $d1 dimensions but series2 has $d2 dimensions"))
        end

        # Compute KL divergence for each dimension
        kl_vec = zeros(d1)
        for i in 1:d1
            kl_vec[i] = _kl_divergence_1d(series1[i, :], series2[i, :], npoints, bandwidth)
        end

        return kl_vec
    end

    throw(ArgumentError("Inputs must be both vectors or both matrices"))
end

"""
    _kl_divergence_1d(x1, x2, npoints, bandwidth)

Internal function to compute KL divergence between two 1D distributions.
"""
function _kl_divergence_1d(x1::AbstractVector, x2::AbstractVector, npoints::Int, bandwidth::Union{Nothing,Real})
    # Estimate PDFs using KDE
    if isnothing(bandwidth)
        kde1 = kde(vec(x1), npoints=npoints)
        kde2 = kde(vec(x2), npoints=npoints)
    else
        kde1 = kde(vec(x1), npoints=npoints, bandwidth=bandwidth)
        kde2 = kde(vec(x2), npoints=npoints, bandwidth=bandwidth)
    end

    # Create common grid spanning both distributions
    x_min = min(minimum(kde1.x), minimum(kde2.x))
    x_max = max(maximum(kde1.x), maximum(kde2.x))
    x_common = range(x_min, x_max, length=npoints)

    # Interpolate PDFs onto common grid
    p1 = _interpolate_pdf(kde1.x, kde1.density, x_common)
    p2 = _interpolate_pdf(kde2.x, kde2.density, x_common)

    # Normalize to ensure they integrate to 1
    dx = step(x_common)
    p1 ./= sum(p1) * dx
    p2 ./= sum(p2) * dx

    # Compute KL divergence: D_KL(p1 || p2) = ∫ p1(x) log(p1(x)/p2(x)) dx
    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-10
    p1_safe = max.(p1, eps)
    p2_safe = max.(p2, eps)

    kl = sum(p1_safe .* log.(p1_safe ./ p2_safe)) * dx

    return kl
end

"""
    _interpolate_pdf(x, density, x_new)

Linear interpolation of PDF values onto new grid.
"""
function _interpolate_pdf(x::AbstractVector, density::AbstractVector, x_new::AbstractRange)
    # Simple linear interpolation
    density_new = zeros(length(x_new))

    for (i, xi) in enumerate(x_new)
        if xi <= x[1]
            density_new[i] = density[1]
        elseif xi >= x[end]
            density_new[i] = density[end]
        else
            # Find bracketing indices
            idx = searchsortedfirst(x, xi)
            if idx > 1 && idx <= length(x)
                # Linear interpolation
                x1, x2 = x[idx-1], x[idx]
                y1, y2 = density[idx-1], density[idx]
                t = (xi - x1) / (x2 - x1)
                density_new[i] = y1 + t * (y2 - y1)
            end
        end
    end

    return density_new
end
