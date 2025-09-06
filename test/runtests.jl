using Test
using Random
using ScoreEstimation

@testset "ScoreEstimation basic API" begin
    Random.seed!(123)
    # Small synthetic dataset: 1D normalized observations
    D = 1
    N = 400
    obs = randn(D, N)
    σ = 0.1

    # KGMM small run (few clusters, quick convergence)
    @testset "KGMM" begin
        res = ScoreEstimation.KGMM(σ, obs; prob=0.05, conv_param=5e-2, i_max=10, show_progress=false)
        @test haskey(res, :centers)
        @test haskey(res, :score)
        @test haskey(res, :divergence)
        @test haskey(res, :counts)
        C = res.Nc
        @test size(res.centers) == (D, C)
        @test size(res.score) == (D, C)
        @test length(res.divergence) == C
        @test length(res.counts) == C
    end

    # Training wrapper (preprocessing=true)
    @testset "Train with KGMM" begin
            nn, losses, _, div_fn, res = ScoreEstimation.train(obs;
            preprocessing=true, σ=σ, neurons=[D, 16, D], n_epochs=3, batch_size=64,
            lr=1e-3, use_gpu=false, verbose=false, kgmm_kwargs=(prob=0.05, conv_param=5e-2, i_max=10, show_progress=false),
            divergence=true, probes=1)
        @test length(losses) == 3
        X = Float32.(res.centers)
        sθ = X -> -nn(X) ./ Float32(σ)
        @test size(sθ(X)) == size(X)
        @test size(div_fn(X)) == (1, size(X,2))
    end

    # Training wrapper (preprocessing=false)
    @testset "Train raw DSM" begin
        nn, losses, _, div_fn, res = ScoreEstimation.train(obs;
            preprocessing=false, σ=σ, neurons=[D, 16, D], n_epochs=3, batch_size=64,
            lr=1e-3, use_gpu=false, verbose=false, divergence=true, probes=1)
        @test length(losses) == 3
        @test res === nothing
        X = Float32.(obs[:, 1:32])
        sθ = X -> -nn(X) ./ Float32(σ)
        @test size(sθ(X)) == size(X)
        @test size(div_fn(X)) == (1, size(X,2))
    end
end
