############### enzyme_fixedsigma_1D_fast_manual_or_enzyme.jl ###############
# ε-net training (predict z), fixed σ≈0.1
# - Same FastSDE data + fixed (X,Z) dataset (Float32)
# - **FAST PATH (default)**: manual backprop with BLAS (mul!) + full preallocation
# - Optional :enzyme backend (loop matmuls inside AD region, with safe scratch)
# - Contiguous minibatches (shuffle once/epoch; views for fast path, buffers for Enzyme)
# - Exact chunked eval; clean @info logging
# - Returns: (ε̂, sθ, θ_flat, Linfo, train_losses, val_losses)

import Pkg
Pkg.activate("."); Pkg.instantiate()

using Random, LinearAlgebra, Statistics, Dates
using FastSDE
using Enzyme

# ---------------- optional DuplicatedNoNeed shim ----------------
const _HAS_DNN = isdefined(Enzyme, :DuplicatedNoNeed)
@inline function _scratch_tag(s, sadj)
    _HAS_DNN ? Enzyme.DuplicatedNoNeed(s, sadj) : Enzyme.Duplicated(s, sadj)
end

# ---------------- 1) Data ----------------
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

# ------------- 2) Fixed ε-dataset (Float32) -------------
"""
    make_fixed_dataset(obs, σ; Ntrain=200_000, Nval=50_000, seed=123)

Precompute (X,Z) with X = y + σ z, Z = z using a fixed RNG.
Returns Float32 matrices: (Xtr, Ztr, Xval, Zval) with shape (D, N*).
"""
function make_fixed_dataset(obs::AbstractMatrix, σ::Real; Ntrain=200_000, Nval=50_000, seed=123)
    T = Float32
    Random.seed!(seed)
    D, N = size(obs)
    obsf = T.(obs); σf = T(σ)

    function draw_block(Nsamples)
        X = Matrix{T}(undef, D, Nsamples)
        Z = Matrix{T}(undef, D, Nsamples)
        @inbounds for j in 1:Nsamples
            idx = rand(1:N)
            @inbounds for i in 1:D
                z = randn(T)
                Z[i,j] = z
                X[i,j] = obsf[i, idx] + σf*z
            end
        end
        X, Z
    end

    Xtr, Ztr = draw_block(Ntrain)
    Xval, Zval = draw_block(Nval)
    return Xtr, Ztr, Xval, Zval
end

# ------------- 3) Model + utilities (no AD deps) -------------
@inline swish!(Y, X) = (@inbounds for j in axes(X,2), i in axes(X,1)
    x = X[i,j]; s = 1f0/(1f0+exp(-x)); Y[i,j] = x*s
end)
@inline swishprime!(D, A) = (@inbounds for j in axes(A,2), i in axes(A,1)
    x = A[i,j]; s = 1f0/(1f0+exp(-x)); D[i,j] = s + x*s*(1f0 - s)
end)

@inline function bias_add!(Y::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds for j in axes(Y,2), i in axes(Y,1)
        Y[i,j] += b[i]
    end
    return Y
end

struct MLP{T}
    W::Vector{Matrix{T}}
    b::Vector{Vector{T}}
    sizes::Vector{Int}
end

# Xavier-uniform init
function init_mlp(in_dim::Int, hidden::AbstractVector{<:Integer}, out_dim::Int;
                  T::Type=Float32, rng=Random.GLOBAL_RNG)
    sizes = Int[in_dim; hidden...; out_dim]
    L = length(sizes)-1
    W = Vector{Matrix{T}}(undef, L)
    b = Vector{Vector{T}}(undef, L)
    for ℓ in 1:L
        nin, nout = sizes[ℓ], sizes[ℓ+1]
        limit = sqrt(T(6)/(T(nin)+T(nout)))
        W[ℓ] = rand(rng, T, nout, nin) .* (T(2)*limit) .- limit
        b[ℓ] = zeros(T, nout)
    end
    return MLP{T}(W, b, sizes)
end

# ---------- FAST PATH forward (BLAS mul! + in-place ops) ----------
# Accepts AbstractMatrix views for Xb and vector-of-(AbstractMatrix) for A/H (so eval views work).
function forward_manual!(model::MLP{T}, Xb::AbstractMatrix{T},
                         A::Vector{<:AbstractMatrix{T}}, H::Vector{<:AbstractMatrix{T}}) where {T}
    L = length(model.W)
    # layer 1
    mul!(A[1], model.W[1], Xb)         # A1 = W1*X
    bias_add!(A[1], model.b[1])
    if L == 1
        copyto!(H[1], A[1])
        return
    else
        swish!(H[1], A[1])             # H1 = swish(A1)
    end
    # hidden
    @inbounds for ℓ in 2:L-1
        mul!(A[ℓ], model.W[ℓ], H[ℓ-1])
        bias_add!(A[ℓ], model.b[ℓ])
        swish!(H[ℓ], A[ℓ])
    end
    # last (identity)
    mul!(A[L], model.W[L], H[L-1])
    bias_add!(A[L], model.b[L])
    copyto!(H[L], A[L])                # output copy for uniformity
    return
end

# ---------- FAST PATH backward (BLAS mul! + fused elementwise) ----------
# Accept AbstractMatrices/vectors so training/eval views or matrices all work.
function backward_manual!(model::MLP{T}, Xb::AbstractMatrix{T}, Zb::AbstractMatrix{T},
                          A::Vector{<:AbstractMatrix{T}}, H::Vector{<:AbstractMatrix{T}},
                          dA::Vector{<:AbstractMatrix{T}}, tmp::Vector{<:AbstractMatrix{T}},
                          gW::Vector{<:AbstractMatrix{T}}, gb::Vector{<:AbstractVector{T}}) where {T}
    L = length(model.W); B = size(Xb,2)
    invB2 = T(2) / T(B)
    # dA_L = invB2*(A_L - Z)
    @inbounds for j in axes(A[L],2), i in axes(A[L],1)
        dA[L][i,j] = invB2*(A[L][i,j] - Zb[i,j])
    end
    # layer L: gW[L] = dA_L * H[L-1]^T, gb[L] = sum(dA_L), dH[L-1] = W[L]^T * dA_L
    fill!(gW[L], zero(T)); fill!(gb[L], zero(T))
    mul!(gW[L], dA[L], transpose(H[L-1]))             # n_out×B * B×n_in → n_out×n_in
    @inbounds for i in axes(dA[L],1), j in axes(dA[L],2)
        gb[L][i] += dA[L][i,j]
    end
    if L > 1
        mul!(tmp[L-1], transpose(model.W[L]), dA[L])  # dH_{L-1} (n_out_{L-1}×B)
    end

    # hidden layers
    @inbounds for ℓ in (L-1):-1:2
        # dA_ℓ = dH_ℓ .* swish'(A_ℓ)
        swishprime!(dA[ℓ], A[ℓ])                      # put σ'(A) in dA[ℓ]
        @inbounds for j in axes(dA[ℓ],2), i in axes(dA[ℓ],1)
            dA[ℓ][i,j] *= tmp[ℓ][i,j]                 # tmp[ℓ] holds dH_ℓ
        end
        # grads
        fill!(gW[ℓ], zero(T)); fill!(gb[ℓ], zero(T))
        mul!(gW[ℓ], dA[ℓ], transpose(H[ℓ-1]))
        @inbounds for i in axes(dA[ℓ],1), j in axes(dA[ℓ],2)
            gb[ℓ][i] += dA[ℓ][i,j]
        end
        # dH_{ℓ-1} = W[ℓ]^T * dA[ℓ]
        mul!(tmp[ℓ-1], transpose(model.W[ℓ]), dA[ℓ])
    end

    # layer 1 (uses dH_1 in tmp[1])
    swishprime!(dA[1], A[1])
    @inbounds for j in axes(dA[1],2), i in axes(dA[1],1)
        dA[1][i,j] *= tmp[1][i,j]
    end
    fill!(gW[1], zero(T)); fill!(gb[1], zero(T))
    mul!(gW[1], dA[1], transpose(Xb))
    @inbounds for i in axes(dA[1],1), j in axes(dA[1],2)
        gb[1][i] += dA[1][i,j]
    end
    return nothing
end

# mean MSE over columns (for eval)
@inline function mse_cols_from_A(A_L::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T<:AbstractFloat}
    return sum(abs2, A_L .- Z) / T(size(A_L,2))
end

# ---------- Adam (in-place) ----------
mutable struct AdamState{T}
    mW::Vector{Matrix{T}}; vW::Vector{Matrix{T}}
    mb::Vector{Vector{T}}; vb::Vector{Vector{T}}
    t::Int; β1::T; β2::T; eps::T
end
function adam_init(model::MLP{T}; β1=0.9, β2=0.999, eps=1f-8) where {T}
    mW = [zeros(T, size(W)) for W in model.W]
    vW = [zeros(T, size(W)) for W in model.W]
    mb = [zeros(T, length(b)) for b in model.b]
    vb = [zeros(T, length(b)) for b in model.b]
    return AdamState{T}(mW, vW, mb, vb, 0, T(β1), T(β2), T(eps))
end

function adam_step!(model::MLP{T}, gW, gb, o::AdamState{T}, η::T) where {T}
    o.t += 1
    β1, β2, eps = o.β1, o.β2, o.eps
    t = o.t
    @inbounds for ℓ in eachindex(model.W)
        # weights
        mW = o.mW[ℓ]; vW = o.vW[ℓ]; W = model.W[ℓ]; g = gW[ℓ]
        @. mW = β1*mW + (1-β1)*g
        @. vW = β2*vW + (1-β2)*g*g
        mhat = mW ./ (1 - β1^t)
        vhat = vW ./ (1 - β2^t)
        @. W = W - η * (mhat ./ (sqrt(vhat) + eps))
        # biases
        mb = o.mb[ℓ]; vb = o.vb[ℓ]; b = model.b[ℓ]; gbℓ = gb[ℓ]
        @. mb = β1*mb + (1-β1)*gbℓ
        @. vb = β2*vb + (1-β2)*gbℓ*gbℓ
        mbhat = mb ./ (1 - β1^t)
        vbhat = vb ./ (1 - β2^t)
        @. b = b - η * (mbhat ./ (sqrt(vbhat) + eps))
    end
    return nothing
end

# ---------- Eval loss in chunks (forward only; BLAS) ----------
function full_loss_chunks(model::MLP{T}, X::Matrix{T}, Z::Matrix{T}; eval_bs::Int=50_000) where {T}
    N = size(X,2); L = length(model.W)
    # scratch sized to eval_bs; reuse
    A = [zeros(T, size(model.W[ℓ],1), eval_bs) for ℓ in 1:L]
    H = [zeros(T, size(model.W[ℓ],1), eval_bs) for ℓ in 1:L]
    acc = zero(T)
    @inbounds @views for off in 1:eval_bs:N
        hi = min(off + eval_bs - 1, N)
        B = hi - off + 1
        Ap = [view(A[ℓ], :, 1:B) for ℓ in 1:L]
        Hp = [view(H[ℓ], :, 1:B) for ℓ in 1:L]
        forward_manual!(model, X[:, off:hi], Ap, Hp)
        acc += sum(abs2, Ap[L] .- Z[:, off:hi])
    end
    return Float64(acc) / N
end

# ---------- Optional Enzyme backend (loop matmuls; safe scratch) ----------
@inline function loop_affine!(Y::AbstractMatrix{T}, W::AbstractMatrix{T},
                              X::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    n_out, n_in = size(W); B = size(X,2)
    @inbounds for j in 1:B, i in 1:n_out
        acc = zero(T)
        @simd for k in 1:n_in
            acc += W[i,k] * X[k,j]
        end
        Y[i,j] = acc + b[i]
    end
    return Y
end

@inline function forward_loops!(A::Vector{<:AbstractMatrix{T}}, H::Vector{<:AbstractMatrix{T}},
                                W::Vector{Matrix{T}}, b::Vector{Vector{T}},
                                X::AbstractMatrix{T}) where {T}
    L = length(W)
    loop_affine!(A[1], W[1], X, b[1]); (L==1) ? copyto!(H[1], A[1]) : swish!(H[1], A[1])
    @inbounds for ℓ in 2:L-1
        loop_affine!(A[ℓ], W[ℓ], H[ℓ-1], b[ℓ]); swish!(H[ℓ], A[ℓ])
    end
    if L >= 2
        loop_affine!(A[L], W[L], H[L-1], b[L]); copyto!(H[L], A[L])
    end
    return H[L]
end

@inline function eps_loss_enzyme(W::Vector{Matrix{T}}, b::Vector{Vector{T}},
                                 X::AbstractMatrix{T}, Z::AbstractMatrix{T},
                                 A::Vector{<:AbstractMatrix{T}}, H::Vector{<:AbstractMatrix{T}}) where {T}
    Y = forward_loops!(A, H, W, b, X)
    return sum(abs2, Y .- Z) / T(size(Z,2))
end

function grad_step_enzyme!(model::MLP{T}, Xb::Matrix{T}, Zb::Matrix{T},
                           A::Vector{Matrix{T}}, H::Vector{Matrix{T}},
                           Aadj::Vector{Matrix{T}}, Hadj::Vector{Matrix{T}},
                           gW::Vector{Matrix{T}}, gb::Vector{Vector{T}},
                           opt::AdamState{T}, η::T) where {T}
    @inbounds for ℓ in eachindex(gW)
        fill!(gW[ℓ], zero(T)); fill!(gb[ℓ], zero(T))
        fill!(Aadj[ℓ], zero(T)); fill!(Hadj[ℓ], zero(T))
    end
    autodiff(Reverse, eps_loss_enzyme,
             Duplicated(model.W, gW),
             Duplicated(model.b, gb),
             Const(Xb), Const(Zb),
             _scratch_tag(A, Aadj),
             _scratch_tag(H, Hadj))
    adam_step!(model, gW, gb, opt, η)
    return nothing
end

# --------- 4) Training loop (same API) ----------
"""
    train_enzyme_eps_fixed(obs; σ=0.1, neurons=[D,128,128,D], Ntrain=200_000, Nval=50_000,
                           batchsize=8192, nepochs=50, lr=1e-3, seed=42,
                           eval_bs=50_000, verbose=true, backend=:manual)

Returns (ε̂, sθ, θ_flat, Linfo, train_losses, val_losses).
Backends:
  :manual (default) → fastest: BLAS forward + manual backprop
  :enzyme          → Enzyme Reverse on loop-based forward (safe & allocation-free)
"""
function train_enzyme_eps_fixed(obs::AbstractMatrix; σ::Real=0.1,
                                neurons::Vector{Int}=[size(obs,1),128,128,size(obs,1)],
                                Ntrain::Int=200_000, Nval::Int=50_000,
                                batchsize::Int=8192, nepochs::Int=50,
                                lr::Float64=1e-3, seed::Int=42,
                                eval_bs::Int=50_000, verbose::Bool=true,
                                backend::Symbol=:manual)

    Random.seed!(seed)
    T = Float32
    D = size(obs,1)
    @assert first(neurons) == D && last(neurons) == D "neurons must be [D, ..., D] with D=$(D)"

    Xtr, Ztr, Xval, Zval = make_fixed_dataset(obs, σ; Ntrain=Ntrain, Nval=Nval, seed=seed)
    steps_per_epoch = (size(Xtr,2) ÷ batchsize)
    @assert steps_per_epoch > 0 "batchsize > Ntrain; reduce batchsize."
    Nuse = steps_per_epoch * batchsize

    model = init_mlp(D, neurons[2:end-1], D; T=T)
    opt   = adam_init(model)
    L = length(model.W)

    # Preallocate training scratch (sized to batchsize)
    A   = [zeros(T, size(model.W[ℓ],1), batchsize) for ℓ in 1:L]
    H   = [zeros(T, size(model.W[ℓ],1), batchsize) for ℓ in 1:L]
    dA  = [zeros(T, size(model.W[ℓ],1), batchsize) for ℓ in 1:L]
    tmp = (L>1) ? [zeros(T, size(model.W[ℓ],1), batchsize) for ℓ in 1:L-1] : Matrix{T}[]

    gW = [zeros(T, size(model.W[ℓ]))     for ℓ in 1:L]
    gb = [zeros(T, length(model.b[ℓ]))   for ℓ in 1:L]

    # Extra scratch for enzyme
    Aadj = [zeros(T, size(A[ℓ])) for ℓ in 1:L]
    Hadj = [zeros(T, size(H[ℓ])) for ℓ in 1:L]

    # Buffers for enzyme batch loading (avoid vector-indexed views)
    Xb_buf = Matrix{T}(undef, D, batchsize)
    Zb_buf = Matrix{T}(undef, D, batchsize)
    @inline function load_batch!(Xbuf::AbstractMatrix{T}, Zbuf::AbstractMatrix{T},
                                 X::AbstractMatrix{T}, Z::AbstractMatrix{T},
                                 idxs::AbstractVector{Int}) where {T}
        D, B = size(Xbuf,1), size(Xbuf,2)
        @inbounds for j in 1:B
            col = idxs[j]
            @simd for i in 1:D
                Xbuf[i,j] = X[i,col]
                Zbuf[i,j] = Z[i,col]
            end
        end
        return Xbuf, Zbuf
    end

    # Warmup JIT on one batch (fast path)
    @views begin
        forward_manual!(model, Xtr[:, 1:batchsize], A, H)
        if L > 1
            backward_manual!(model, Xtr[:, 1:batchsize], Ztr[:, 1:batchsize], A, H, dA, tmp, gW, gb)
        else
            B = batchsize; invB2 = T(2)/T(B)
            @inbounds for j in 1:B, i in axes(A[1],1)
                dA[1][i,j] = invB2*(A[1][i,j] - Ztr[i,j])
            end
            fill!(gW[1], zero(T)); fill!(gb[1], zero(T))
            mul!(gW[1], dA[1], transpose(Xtr[:,1:batchsize]))
            @inbounds for i in axes(dA[1],1), j in 1:B
                gb[1][i] += dA[1][i,j]
            end
        end
        adam_step!(model, gW, gb, opt, T(lr))
    end

    train_losses = Vector{Float64}(undef, nepochs)
    val_losses   = Vector{Float64}(undef, nepochs)

    for e in 1:nepochs
        t0 = time()

        # Shuffle once → contiguous minibatches
        p = randperm(size(Xtr,2))

        if backend === :manual
            @inbounds @views begin
                Xs = Xtr[:, p];  Zs = Ztr[:, p]
                for lo in 1:batchsize:Nuse
                    hi = lo + batchsize - 1
                    Xb = Xs[:, lo:hi]; Zb = Zs[:, lo:hi]
                    forward_manual!(model, Xb, A, H)
                    if L > 1
                        backward_manual!(model, Xb, Zb, A, H, dA, tmp, gW, gb)
                    else
                        B = batchsize; invB2 = T(2)/T(B)
                        @inbounds for j in 1:B, i in axes(A[1],1)
                            dA[1][i,j] = invB2*(A[1][i,j] - Zb[i,j])
                        end
                        fill!(gW[1], zero(T)); fill!(gb[1], zero(T))
                        mul!(gW[1], dA[1], transpose(Xb))
                        @inbounds for i in axes(dA[1],1), j in 1:B
                            gb[1][i] += dA[1][i,j]
                        end
                    end
                    adam_step!(model, gW, gb, opt, T(lr))
                end
            end
        elseif backend === :enzyme
            base = @view p[1:Nuse]  # drop tail for constant shapes
            @inbounds for s in 1:steps_per_epoch
                lo = (s-1)*batchsize + 1; hi = s*batchsize
                b  = @view base[lo:hi]
                load_batch!(Xb_buf, Zb_buf, Xtr, Ztr, b)
                grad_step_enzyme!(model, Xb_buf, Zb_buf, A, H, Aadj, Hadj, gW, gb, opt, T(lr))
            end
        else
            error("Unknown backend = $backend. Use :manual or :enzyme.")
        end

        # metrics
        train_losses[e] = full_loss_chunks(model, Xtr, Ztr; eval_bs=eval_bs)
        val_losses[e]   = full_loss_chunks(model, Xval, Zval; eval_bs=eval_bs)
        verbose && @info "epoch stats" epoch=e train=train_losses[e] val=val_losses[e] secs=(time()-t0)
    end

    # ε̂ and score sθ (forward-only; BLAS)
    ε̂ = let model=model, L=L
        function (X::AbstractMatrix{<:Real})
            Tθ = eltype(model.W[1])
            Xf = Tθ.(X)
            B  = size(Xf,2)
            Aeval = [zeros(Tθ, size(model.W[ℓ],1), B) for ℓ in 1:L]
            Heval = [zeros(Tθ, size(model.W[ℓ],1), B) for ℓ in 1:L]
            forward_manual!(model, Xf, Aeval, Heval)
            return Heval[L]
        end
    end
    sθ = let ε̂=ε̂
        (X, σ::Real) -> -(ε̂(X)) ./ Float32(σ)
    end

    # Flatten θ
    θ_flat = vcat([vec(W) for W in model.W]..., [copy(b) for b in model.b]...)

    # "L" info compatible placeholder (sizes etc.)
    Linfo = (sizes = model.sizes,)

    return ε̂, sθ, θ_flat, Linfo, train_losses, val_losses
end

# ---------------- 5) Run it ----------------
const D = size(obs,1); @assert D == 1
σ       = 0.10
neurons = [D, 100, 50, D]
Ntrain, Nval = 200_000, 50_000
batchsize     = 8192
nepochs       = 50
lr            = 1e-3
seed          = 42
eval_bs       = 50_000

@info "Starting ε-training on FIXED dataset (σ=$(σ))..."
ε̂, sθ, θ, L, train_losses, val_losses =
    train_enzyme_eps_fixed(obs; σ=σ, neurons=neurons, Ntrain=Ntrain, Nval=Nval,
                           batchsize=batchsize, nepochs=nepochs, lr=lr,
                           seed=seed, eval_bs=eval_bs, verbose=true,
                           backend=:manual)  # switch to :enzyme to test Enzyme path
@info "Done. Final train=$(last(train_losses))  val=$(last(val_losses))"

# Small score table (1D)
xs = collect(range(-3f0, 3f0, length=21))
Xg = reshape(Float32.(xs), 1, :)
Sg = sθ(Xg, σ)
@info "Sample scores (x, sθ(x)):"; for i in eachindex(xs)
    @info("x=$(round(xs[i], digits=3))  sθ=$(round(Sg[1,i], digits=5))")
end
##################################################################
##

# ---------------- 6) Plots: losses and scores ----------------
begin
    # Load Plots robustly
    try
        using Plots
    catch
        import Pkg
        Pkg.add("Plots")
        using Plots
    end

    # ---- (a) Loss curves ----
    epochs = collect(1:length(train_losses))
    p_loss = plot(epochs, train_losses; label="train", xlabel="epoch",
                  ylabel="mean MSE", title="ε-net loss", lw=2)
    plot!(p_loss, epochs, val_losses; label="val", lw=2, ls=:dash)

    # ---- (b) True vs learned score on a grid ----
    # Empirical "true" score via k-NN regression over the fixed (X,Z) dataset
    function empirical_true_score_1d(X::Matrix{Float32}, Z::Matrix{Float32},
                                     σ::Float32, xs::AbstractVector{<:Real}; K::Int=5000)
        xv = vec(X); zv = vec(Z)
        p  = sortperm(xv)
        xsrt = xv[p]; zsrt = zv[p]
        n = length(xsrt)
        s_true = similar(Float32.(xs))
        @inbounds for (t, xg0) in enumerate(xs)
            xg  = Float32(xg0)
            pos = searchsortedfirst(xsrt, xg)
            lo  = max(1, pos - K ÷ 2)
            hi  = min(n, lo + K - 1)
            lo  = max(1, hi - K + 1)  # ensure window size K when possible
            s_true[t] = -mean(view(zsrt, lo:hi)) / σ
        end
        return s_true
    end

    # Grid and learned score
    xs = collect(range(-3f0, 3f0, length=201))
    Xg = reshape(Float32.(xs), 1, :)
    Sg = sθ(Xg, σ)  # learned score

    # Recreate the fixed dataset (same seed) to estimate "true" score
    Xtr_p, Ztr_p, Xval_p, Zval_p = make_fixed_dataset(obs, σ; Ntrain=Ntrain, Nval=Nval, seed=seed)
    Xall = hcat(Xtr_p, Xval_p); Zall = hcat(Ztr_p, Zval_p)
    s_true = empirical_true_score_1d(Xall, Zall, Float32(σ), xs; K=5000)

    p_score = plot(xs, vec(Sg[1, :]); label="learned sθ(x)", lw=3,
                   xlabel="x", ylabel="score", title="True vs learned scores")
    plot!(p_score, xs, s_true; label="empirical true s*(x)", lw=3, ls=:dash)

    # Show side-by-side and save
    plt = plot(p_loss, p_score; layout=(1,2), size=(1200,400))
    savefig(plt, "loss_and_scores.png")
    @info "Saved plots to loss_and_scores.png"
end
