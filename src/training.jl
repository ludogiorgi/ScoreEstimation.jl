############################
#  training.jl (optimized) #
############################

# This file provides an optimized, time-independent ε-training pipeline.
# Key points:
# - Removes all diffusion-time dependencies: σ is a scalar only.
# - Vectorized data generation with Float32 tensors.
# - Device-aware training loops (CPU/GPU) with on-device DataLoaders.
# - Professional API with a high-level wrapper `train(...; preprocessing=...)`.
# - Optional preprocessing uses KGMM outputs for supervised interpolation.


############################
#  Data (fixed σ only)    #
############################

"""
    _generate_xz(Y::AbstractMatrix, σ::Real)

Vectorized draw of ε ~ N(0, I) and X = Y + σ·ε.
Input `Y` is shape (D, N); returns `(X, Z)` with same shape, Float32.
"""
function _generate_xz(Y::AbstractMatrix, σ::Real)
    D, N = size(Y)
    T = Float32
    Z = randn(T, D, N)
    Yf = (eltype(Y) === T) ? Y : T.(Y)
    X = @views Yf .+ T(σ) .* Z
    return X, Z
end

"""
    _generate_data(obs, σ::Real)

Generate inputs and targets for fixed noise level: X = Y + σ·ε, Z = ε.
`obs` is (D, N). Returns `(X, Z)` as Float32 matrices (D, N).
"""
function _generate_data(obs, σ::Real)
    return _generate_xz(obs, σ)
end


############################
#  Model & losses          #
############################

"""
    _swish(x)

Swish activation: x .* σ(x)
"""
_swish(x) = x .* Flux.sigmoid.(x)

"""
    _build_mlp(neurons::Vector{Int}; activation=_swish, last_activation=identity)

Create an MLP with architecture defined by `neurons` (full layer sizes including
input and output). Internal helper.
"""
function _build_mlp(neurons::Vector{Int}; activation=_swish, last_activation=identity)
    layers = Vector{Any}(undef, length(neurons) - 1)
    for i in 1:length(neurons)-2
        layers[i] = Flux.Dense(neurons[i], neurons[i+1], activation)
    end
    layers[end] = Flux.Dense(neurons[end-1], neurons[end], last_activation)
    return Flux.Chain(layers...)
end

"""
    _build_mlp_from_hidden(D::Integer, hidden::Vector{Int}; activation=_swish, last_activation=identity)

Create an MLP with input/output size `D` and hidden layer sizes given by `hidden`.
User-facing APIs should pass only `hidden` sizes; the first and last layer sizes
are inferred as `D`.
"""
function _build_mlp_from_hidden(D::Integer, hidden::Vector{Int}; activation=_swish, last_activation=identity)
    full = isempty(hidden) ? [D, D] : vcat(D, hidden, D)
    return _build_mlp(full; activation=activation, last_activation=last_activation)
end

"""
    _loss_mse(nn, inputs, targets)

Compute mean squared error between predictions and targets.
"""
function _loss_mse(nn, inputs, targets)
    Flux.mse(nn(inputs), targets)
end

# Progress bar handled by ProgressMeter.jl when verbose=true

"""
    _weighted_mse(Ŷ, Y, w)

Weighted MSE across batch columns using non-negative weights `w` (length B).
Returns sum_j w_j * ||Ŷ[:,j] - Y[:,j]||^2 / (sum(w) * D).
"""
function _weighted_mse(Ŷ::AbstractMatrix, Y::AbstractMatrix, w::AbstractVector)
    @assert size(Ŷ) == size(Y)
    @assert size(Y,2) == length(w)
    D = size(Y,1)
    col = sum(abs2, Ŷ .- Y; dims=1)
    return (sum(col .* reshape(w, 1, :)) / (sum(w) * D)) |> eltype(Y)
end


############################
#  Training loops (Flux)   #
############################

"""
    train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ::Real; kwargs...)

Train a new ε-network (predicts z) with fixed σ.

Note: `neurons` contains only the hidden layer sizes. The input and output
layer sizes are automatically inferred as the system dimension `D = size(obs,1)`.
"""
function train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ::Real; 
               opt=Flux.Adam(0.001), activation=_swish, last_activation=identity,
               use_gpu=true, verbose=false)

    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    D = size(obs, 1)
    nn = _build_mlp_from_hidden(D, neurons; activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)

    losses = Float32[]
    obsf = Float32.(obs)
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        X, Z = _generate_data(obsf, σ)
        Xd, Zd = device(X), device(Z)
        _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])  # warmup
        loader = Flux.DataLoader((Xd, Zd), batchsize=batch_size, shuffle=true)

        ep = zero(Float32)
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)

    return nn, losses
end

############################
#  Divergence from score   #
############################

"""
    _divergence_hutchinson(nn, X; probes=1, rademacher=true)

Estimate ∇·f(X) for f = nn (ε-predictor) via Hutchinson's estimator.
Input `X` is (D, B). Returns a (1, B) array with divergence per column.
Works on CPU/GPU depending on `X` device.
"""
function _divergence_hutchinson(nn::Flux.Chain, X::AbstractMatrix;
                               probes::Integer=1, rademacher::Bool=true)
    T = eltype(X)
    B = size(X, 2)
    div = similar(X, T, 1, B); fill!(div, zero(T))
    for _ in 1:probes
        # Allocate V to match output shape (assume D_out == D_in)
        V = similar(X)
        if rademacher
            rand!(V)
            @. V = ifelse(V > T(0.5), one(T), -one(T))
        else
            randn!(V)
        end
        # gradient of f(X) = sum(nn(X) .* V) w.r.t X
        gX = Flux.gradient(x -> sum(nn(x) .* V), X)[1]
        # accumulate vᵀ J v per sample (column-wise)
        div .+= sum(V .* gX; dims=1)
    end
    div ./= T(probes)
    return div
end

"""
    _divergence_from_eps(nn, X; σ, probes=1, rademacher=true)

Compute ∇·sθ(X) when sθ(X) = -(1/σ) ε̂(X): returns −(1/σ) ∇·ε̂.
"""
function _divergence_from_eps(nn::Flux.Chain, X::AbstractMatrix;
                              σ::Real, probes::Integer=1, rademacher::Bool=true)
    d_eps = _divergence_hutchinson(nn, X; probes=probes, rademacher=rademacher)
    return @. -(d_eps) / Float32(σ)
end

############################
#  Jacobian of the score   #
############################

"""
    _jacobian_from_eps(nn, X) -> Jε

Compute the exact Jacobian of the ε-predictor `nn` at batched inputs `X`.
Returns a 3D array with shape `(D_out, D_in, B)` for each column in `X`.

Implementation uses reverse-mode accumulation of rows via
`gX = ∇_X sum(nn(X) .* V)` with `V[i, :] .= 1`, which yields row `i` of the
Jacobian for all batch columns at once. Works on CPU and GPU.
"""
function _jacobian_from_eps(nn::Flux.Chain, X::AbstractMatrix)
    T = eltype(X)
    D, B = size(X)
    J = similar(X, T, D, D, B)  # (D_out, D_in, B)
    # Accumulate rows i = 1..D across the whole batch in D reverse-mode passes
    for i in 1:D
        V = zero(X)
        @views V[i, :] .= one(T)
        gX = Flux.gradient(x -> sum(nn(x) .* V), X)[1]  # gX[:, j] = (J_j)' * e_i = row i of J_j
        @inbounds for j in 1:B
            @views J[i, :, j] .= gX[:, j]
        end
    end
    return J
end

"""
    _jacobian_score(nn, X; σ) -> J_s

Return the Jacobian of the score s(x) = -(1/σ) ε̂(x) at batched inputs `X`.
Shape: `(D, D, B)` where `B` is the number of columns in `X`.
"""
function _jacobian_score(nn::Flux.Chain, X::AbstractMatrix; σ::Real)
    Jε = _jacobian_from_eps(nn, X)
    @. Jε = -(Jε) / Float32(σ)
    return Jε
end

"""
    train(obs, n_epochs, batch_size, nn::Chain, σ::Real; kwargs...)

Continue training an existing ε-network with fixed σ.
"""
function train(obs, n_epochs, batch_size, nn::Chain, σ::Real; 
               opt=Flux.Adam(0.001), use_gpu=true, verbose=false)

    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    nn = nn |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    losses = Float32[]

    obsf = Float32.(obs)
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        X, Z = _generate_data(obsf, σ)
        Xd, Zd = device(X), device(Z)
        _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
        loader = Flux.DataLoader((Xd, Zd), batchsize=batch_size, shuffle=true)
        ep = zero(Float32)
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)
    return nn, losses
end

"""
    train(obs_tuple, n_epochs, batch_size, neurons; kwargs...)

Train on a precomputed dataset `(X, Z)` (e.g., from KGMM centers).

Note: `neurons` contains only the hidden layer sizes. The first and last layer
sizes are inferred from `size(X,1)`.
"""
function train(obs_tuple, n_epochs, batch_size, neurons; 
               opt=Flux.Adam(0.001), activation=_swish, last_activation=identity,
               use_gpu=true, verbose=false)

    X, Z = obs_tuple
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    D = size(X, 1)
    nn = _build_mlp_from_hidden(D, neurons; activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        ep = zero(Float32)
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)

    return nn, losses
end

"""
    train((X, Z), n_epochs, batch_size, nn::Chain; kwargs...)

Continue training an existing ε-network on a precomputed dataset `(X, Z)`.
"""
function train(obs_tuple::Tuple{<:AbstractMatrix,<:AbstractMatrix},
               n_epochs::Integer,
               batch_size::Integer,
               nn::Chain;
               opt=Flux.Adam(0.001), use_gpu=true, verbose=false)

    X, Z = obs_tuple
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    nn = nn |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        ep = zero(Float32)
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)

    return nn, losses
end

"""
    train((X, Z, w), n_epochs, batch_size, neurons; ...)

Train on a precomputed weighted dataset where `w` are per-sample weights
(e.g., KGMM cluster counts). Uses weighted MSE.

Note: `neurons` contains only the hidden layer sizes. The first and last layer
sizes are inferred from `size(X,1)`.
"""
function train(obs_tuple::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractVector},
               n_epochs::Integer,
               batch_size::Integer,
               neurons;
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
               use_gpu=true, verbose=false)

    X, Z, w = obs_tuple
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    D = size(X, 1)
    nn = _build_mlp_from_hidden(D, neurons; activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    wd = device(Float32.(w))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd, wd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        ep = zero(Float32)
        for (Xb, Zb, wb) in loader
            loss, grads = Flux.withgradient(nn) do m
                _weighted_mse(m(Xb), Zb, vec(wb))
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)

    return nn, losses
end

"""
    train((X, Z, w), n_epochs, batch_size, nn::Chain; ...)

Continue training an existing ε-network on a precomputed weighted dataset
where `w` are per-sample weights (e.g., KGMM cluster counts). Uses weighted MSE.
"""
function train(obs_tuple::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractVector},
               n_epochs::Integer,
               batch_size::Integer,
               nn::Chain;
               opt=Flux.Adam(0.001), use_gpu=true, verbose=false)

    X, Z, w = obs_tuple
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    nn = nn |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    wd = device(Float32.(w))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd, wd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    verbose && (p = Progress(n_epochs; desc="Epochs", showspeed=false))
    for e in 1:n_epochs
        ep = zero(Float32)
        for (Xb, Zb, wb) in loader
            loss, grads = Flux.withgradient(nn) do m
                _weighted_mse(m(Xb), Zb, vec(wb))
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
        verbose && next!(p)
    end
    verbose && finish!(p)

    return nn, losses
end

"""
    _check_loss(obs, nn, σ::Real; n_samples=1, verbose=false)

Evaluate average MSE over random draws X = Y + σ·ε.
"""
function _check_loss(obs, nn, σ::Real; n_samples=1, verbose=false)
    loss = 0.0
    obsf = Float32.(obs)
    for _ in 1:n_samples
        X, Z = _generate_data(obsf, σ)
        loss += _loss_mse(nn, X, Z)
    end
    return loss / n_samples
end


############################
#  KGMM integration        #
############################

"""
    _dataset_from_kgmm(res::NamedTuple, σ::Real)

Build a supervised (X, Z) dataset from KGMM result `res`.
Using s(x) = -(1/σ) E[z|x] ⇒ E[z|x] = -σ s(x), so set targets Z = -σ·s.
"""
function _dataset_from_kgmm(res::NamedTuple, σ::Real)
    X = Float32.(res.centers)
    S = Float32.(res.score)
    Z = @. (-Float32(σ)) * S
    return X, Z
end

"""
    train(obs; preprocessing=false, σ=0.1, neurons=[128,128],
          n_epochs=50, batch_size=8192, lr=1e-2, use_gpu=false, verbose=true,
          kgmm_kwargs=(;), nn=nothing)

High-level wrapper:
- preprocessing=false: ε-training from random draws X=Y+σ·ε.
- preprocessing=true : run KGMM(σ, obs; kgmm_kwargs...) and supervise on centroids.
Returns `(nn, train_losses, val_losses, div_fn, kgmm)`; `val_losses` is empty in this wrapper.
`kgmm` is the output of KGMM(σ, obs; ...) when `preprocessing=true`, otherwise `nothing`.
"""
function train(obs::AbstractMatrix; preprocessing::Bool=false,
               σ::Real=0.1,
               neurons::Vector{Int}=[128,128],
               n_epochs::Int=50,
               batch_size::Int=8192,
               lr::Float64=1e-2,
               use_gpu::Bool=false,
               verbose::Bool=true,
               kgmm_kwargs=(;),
               divergence::Bool=false,
               probes::Integer=1,
               rademacher::Bool=true,
               jacobian::Bool=false,
               nn::Union{Flux.Chain,Nothing}=nothing)

    opt = Flux.Adam(Float32(lr))
    device = (use_gpu && CUDA.functional()) ? gpu : cpu

    if !preprocessing
        if nn === nothing
            nn, losses = train(obs, n_epochs, batch_size, neurons, σ; opt=opt, use_gpu=use_gpu, verbose=verbose)
        else
            if verbose && !isempty(neurons)
                println("Existing nn provided; ignoring neurons and continuing training.")
            end
            nn, losses = train(obs, n_epochs, batch_size, nn, σ; opt=opt, use_gpu=use_gpu, verbose=verbose)
        end
        div_fn = if divergence
            let nn=nn, device=device, probes=probes, rademacher=rademacher, σ=σ
                X -> begin
                    Xd = device(Float32.(X))
                    Array(_divergence_from_eps(nn, Xd; σ=σ, probes=probes, rademacher=rademacher))
                end
            end
        else
            X -> zeros(eltype(X), 1, size(X,2))
        end
        jac_fn = if jacobian
            let nn=nn, device=device, σ=σ
                X -> begin
                    Xd = device(Float32.(X))
                    Array(_jacobian_score(nn, Xd; σ=σ))
                end
            end
        else
            X -> zeros(eltype(X), size(X,1), size(X,1), size(X,2))
        end
        return nn, losses, Float32[], div_fn, jac_fn, nothing
    else
        # Compute KGMM via the project module; expects `ScoreEstimation` to be in scope.
        res = ScoreEstimation.KGMM(σ, obs; kgmm_kwargs...)
        X, Z = _dataset_from_kgmm(res, σ)
        w    = Float32.(res.counts)
        if nn === nothing
            nn, losses = train((X, Z, w), n_epochs, batch_size, neurons; opt=opt, use_gpu=use_gpu, verbose=verbose)
        else
            if verbose && !isempty(neurons)
                println("Existing nn provided; ignoring neurons and continuing training on KGMM dataset.")
            end
            nn, losses = train((X, Z, w), n_epochs, batch_size, nn; opt=opt, use_gpu=use_gpu, verbose=verbose)
        end
        div_fn = if divergence
            let nn=nn, device=device, probes=probes, rademacher=rademacher, σ=σ
                Xq -> begin
                    Xd = device(Float32.(Xq))
                    Array(_divergence_from_eps(nn, Xd; σ=σ, probes=probes, rademacher=rademacher))
                end
            end
        else
            Xq -> zeros(eltype(Xq), 1, size(Xq,2))
        end
        jac_fn = if jacobian
            let nn=nn, device=device, σ=σ
                Xq -> begin
                    Xd = device(Float32.(Xq))
                    Array(_jacobian_score(nn, Xd; σ=σ))
                end
            end
        else
            Xq -> zeros(eltype(Xq), size(Xq,1), size(Xq,1), size(Xq,2))
        end
        return nn, losses, Float32[], div_fn, jac_fn, res
    end
end
