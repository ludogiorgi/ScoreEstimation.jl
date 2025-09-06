################ flux_eps_fixedsigma_1D.jl ################
# Flux baseline for ε-training (predict z), fixed σ≈0.1
# - 1D FastSDE data generation + normalization
# - Precomputed fixed (X,Z) dataset (same as Enzyme version)
# - Flux training loop with Adam; logs epoch time + train/val ε-loss
# - Score function available as sθ(x) = -ε̂(x)/σ

import Pkg
Pkg.activate("."); Pkg.instantiate()


using Random, LinearAlgebra, Statistics, Printf, Dates
using Flux
using FastSDE
using CUDA

# =========================== 1) Data ===========================
const params = (a=-0.0222, b=-0.2, c=0.0494, F_tilde=0.6, s=0.7071)
const dt = 0.01
const Nsteps = 10_000
const u0 = [0.0]
const resolution = 10
const n_ens = 100

function drift!(du, u, t)
    du[1] = params.F_tilde + params.a*u[1] + params.b*u[1]^2 - params.c*u[1]^3
end
sigma!(du, u, t) = (du[1] = params.s)

obs_nn = evolve(u0, dt, Nsteps, drift!, sigma!;
                timestepper=:euler, resolution=resolution,
                sigma_inplace=true, n_ens=n_ens)
@info "FastSDE trajectory shape: $(size(obs_nn))"

M = mean(obs_nn, dims=2); S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S
@info "Normalized data shape: $(size(obs))"

# ====================== 2) Fixed ε-dataset ======================
"""
    make_fixed_dataset(obs, σ; Ntrain=200_000, Nval=50_000, seed=123)

Precompute (X,Z) with X = y + σ z, Z = z using a fixed RNG.
Returns Float32 matrices: (Xtr, Ztr, Xval, Zval) with shape (D, N*).
Optimized to avoid scalar loops and leverage vectorized sampling.
"""
function make_fixed_dataset(obs::AbstractMatrix, σ::Real; Ntrain=200_000, Nval=50_000, seed=123)
    T = Float32
    Random.seed!(seed)
    D, N = size(obs)
    obsf = T.(obs); σf = T(σ)

    draw_block(Nsamples) = begin
        idx = rand(1:N, Nsamples)
        Z = randn(T, D, Nsamples)
        X = @views obsf[:, idx] .+ σf .* Z
        X, Z
    end

    Xtr, Ztr = draw_block(Ntrain)
    Xval, Zval = draw_block(Nval)
    return Xtr, Ztr, Xval, Zval
end

# ===================== 3) Flux model & utils =====================
# swish(x) = x * sigmoid(x)
swish(x) = x .* Flux.sigmoid.(x)

"""
    create_nn(neurons; activation=swish, last_activation=identity)

Build a Flux MLP Chain mapping D → D, matching your original API.
`neurons` is e.g. [D, 128, 128, D].
"""
function create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)
    layers = Any[]
    for i in 1:length(neurons)-2
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    push!(layers, Flux.Dense(neurons[end-1], neurons[end], last_activation))
    return Flux.Chain(layers...)
end

# exact chunked loss over a full dataset to avoid GPU OOM
function full_loss(nn, X::AbstractMatrix, Z::AbstractMatrix; eval_bs::Int=65_536)
    N = size(X,2); acc = 0.0f0
    off = 1
    while off <= N
        hi = min(off + eval_bs - 1, N)
        Xb = @view X[:, off:hi]
        Zb = @view Z[:, off:hi]
        acc += Flux.mse(nn(Xb), Zb) * Float32(hi - off + 1)
        off = hi + 1
    end
    return Float32(acc / N)
end

# ===================== 4) Train (fixed dataset) ==================
"""
    train_flux_eps_fixed(obs; σ=0.1, neurons=[D,128,128,D], Ntrain=200_000, Nval=50_000,
                         batchsize=8192, nepochs=50, lr=1e-2, seed=42, use_gpu=true, verbose=true)

Trains ε-net on a precomputed fixed dataset. Returns (ε̂, sθ, nn, train_losses, val_losses).
"""
function train_flux_eps_fixed(obs::AbstractMatrix; σ::Real=0.1,
                              neurons::Vector{Int}=[size(obs,1),128,128,size(obs,1)],
                              Ntrain::Int=200_000, Nval::Int=50_000,
                              batchsize::Int=8192, nepochs::Int=50,
                              lr::Float64=1e-2, seed::Int=42,
                              use_gpu::Bool=false, verbose::Bool=true)

    Random.seed!(seed)
    T = Float32
    D = size(obs,1)
    @assert first(neurons)==D && last(neurons)==D "neurons must be [D, ..., D] with D=$(D)"

    # fixed dataset
    Xtr, Ztr, Xval, Zval = make_fixed_dataset(obs, σ; Ntrain=Ntrain, Nval=Nval, seed=seed)
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    devname = (device === gpu ? "GPU" : "CPU")
    verbose && @info "Using $devname"

    # model & optimizer
    nn = create_nn(neurons) |> device |> f32
    opt = Flux.setup(Flux.Adam(Float32(lr)), nn)

    # Move datasets to device once to avoid per-batch transfers
    Xtr_d, Ztr_d = device(Xtr), device(Ztr)
    Xval_d, Zval_d = device(Xval), device(Zval)

    # one warmup forward for JIT
    _ = nn(@view Xtr_d[:, 1:min(batchsize, size(Xtr_d,2))])

    # data loader on-device to eliminate inner-loop transfers
    loader = Flux.DataLoader( (Xtr_d, Ztr_d), batchsize=batchsize, shuffle=true )

    train_losses = Float32[]; val_losses = Float32[]
    for epoch in 1:nepochs
        t0 = time()
        # epoch loop
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt, nn, grads[1])
        end
        # exact metrics on full datasets (chunked)
        tr = full_loss(nn, Xtr_d, Ztr_d)
        vl = full_loss(nn, Xval_d, Zval_d)
        push!(train_losses, tr); push!(val_losses, vl)
        dt = time() - t0
        verbose && @info(@sprintf "epoch=%3d  train=%.6f  val=%.6f  (%.2fs)" epoch tr vl dt)
    end

    # ε̂ and score wrappers
    epŝ = let nn=nn, device=device
        X -> cpu(nn(device(f32(X))))
    end
    sθ = let epŝ=epŝ, σf=T(σ)
        X -> -epŝ(X) ./ σf
    end
    return epŝ, sθ, nn, train_losses, val_losses
end

# ======================== 5) Run it ==============================
const D = size(obs,1); @assert D == 1
σ = 0.10
neurons = [D, 100, 50, D]
Ntrain, Nval = 200_000, 50_000
batchsize = 8192
nepochs   = 50
lr        = 1e-2
seed      = 42
use_gpu   = true   # set false to force CPU

@info "Starting Flux ε-training on FIXED dataset (σ=$(σ))..."
@time epŝ, sθ, nn, train_losses, val_losses =
    train_flux_eps_fixed(obs; σ=σ, neurons=neurons, Ntrain=Ntrain, Nval=Nval,
                         batchsize=batchsize, nepochs=nepochs, lr=lr,
                         seed=seed, use_gpu=use_gpu, verbose=true)
@info "Done. Final train=$(last(train_losses))  val=$(last(val_losses))"

# Small score table (1D), same as Enzyme script
xs = collect(range(-3f0, 3f0, length=21))
Xg = reshape(Float32.(xs), 1, :)
Sg = sθ(Xg)
@info "Sample scores (x, sθ(x)):"
for i in eachindex(xs)
    @info(@sprintf "x=% 7.3f  sθ= % 9.5f" xs[i] Sg[1,i])
end
###############################################################
