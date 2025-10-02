cd(@__DIR__)
using Pkg
Pkg.activate("..")
Pkg.instantiate()

using ScoreEstimation
using Random
using Statistics

println("Testing relative_entropy function")
println("="^50)

# Set seed for reproducibility
Random.seed!(42)

# Test 1: 1D case with identical distributions (should be ~0)
println("\nTest 1: 1D - Identical distributions")
x1 = randn(5000)
kl_identical = relative_entropy(x1, x1)
println("  D_KL(x || x) = $kl_identical (expected ≈ 0)")

# Test 2: 1D case with shifted distributions
println("\nTest 2: 1D - Shifted distributions")
x2 = randn(5000)
x3 = randn(5000) .+ 1.0  # shifted by 1
kl_shifted = relative_entropy(x2, x3)
println("  D_KL(N(0,1) || N(1,1)) = $kl_shifted")

# Test 3: 1D case with different variances
println("\nTest 3: 1D - Different variances")
x4 = randn(5000)
x5 = randn(5000) .* 2.0  # doubled variance
kl_variance = relative_entropy(x4, x5)
println("  D_KL(N(0,1) || N(0,4)) = $kl_variance")

# Test 4: Multi-dimensional case (2D)
println("\nTest 4: 2D - Multi-dimensional")
X1 = randn(2, 5000)
X2 = randn(2, 5000) .+ [0.5, 1.0]
kl_2d = relative_entropy(X1, X2)
println("  D_KL(X1 || X2) = $kl_2d")
println("  (Vector with KL divergence for each dimension)")

# Test 5: Multi-dimensional case (3D)
println("\nTest 5: 3D - Multi-dimensional")
X3 = randn(3, 3000)
X4 = randn(3, 3000) .+ [0.3, 0.5, 0.2]
kl_3d = relative_entropy(X3, X4)
println("  D_KL(X3 || X4) = $kl_3d")
println("  (Vector with KL divergence for each dimension)")

# Test 6: Asymmetry of KL divergence
println("\nTest 6: Asymmetry test")
x6 = randn(5000)
x7 = randn(5000) .+ 0.5
kl_forward = relative_entropy(x6, x7)
kl_reverse = relative_entropy(x7, x6)
println("  D_KL(x6 || x7) = $kl_forward")
println("  D_KL(x7 || x6) = $kl_reverse")
println("  KL divergence is asymmetric: D_KL(P || Q) ≠ D_KL(Q || P)")

println("\n" * "="^50)
println("All tests completed successfully!")
