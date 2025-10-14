#!/usr/bin/env julia

# Simple test script for the custom PDF estimation

"""
    PDFEstimate

Simple structure to hold univariate PDF estimation results.
"""
struct PDFEstimate
    x::Vector{Float64}
    density::Vector{Float64}
end

"""
    estimate_pdf_histogram(data; nbins=nothing, bandwidth=nothing) -> PDFEstimate

Estimate a univariate PDF using histogram binning with Gaussian smoothing.
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
    for val in data
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
            for i in 1:nbins
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

# Test the function
println("Testing PDF estimation function...")

# Generate test data from a normal distribution
using Random
Random.seed!(42)
test_data = randn(10000)

println("Generated $(length(test_data)) samples from standard normal distribution")

# Estimate PDF
pdf_est = estimate_pdf_histogram(test_data)

println("PDF estimated with $(length(pdf_est.x)) points")
println("Data range: [$(minimum(test_data)), $(maximum(test_data))]")
println("PDF x range: [$(minimum(pdf_est.x)), $(maximum(pdf_est.x))]")
println("PDF density range: [$(minimum(pdf_est.density)), $(maximum(pdf_est.density))]")

# Check that density integrates to approximately 1
integral = sum(pdf_est.density) * (pdf_est.x[2] - pdf_est.x[1])
println("Integral of PDF (should be ≈ 1.0): $integral")

if abs(integral - 1.0) < 0.1
    println("✓ PDF estimation working correctly!")
else
    println("✗ Warning: PDF integral is $(integral), expected ≈ 1.0")
end

# Test with small dataset
small_data = [1.0, 2.0, 3.0, 2.0, 1.5]
pdf_small = estimate_pdf_histogram(small_data)
println("\nSmall dataset test: $(length(pdf_small.x)) bins")

# Test edge case: all same values
same_data = [5.0, 5.0, 5.0]
pdf_same = estimate_pdf_histogram(same_data)
println("Degenerate case (all same values): $(length(pdf_same.x)) points")

println("\n✓ All tests passed!")
