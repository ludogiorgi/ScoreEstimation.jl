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
    generate_xz(Y::AbstractMatrix, σ::Real)

Vectorized draw of ε ~ N(0, I) and X = Y + σ·ε.
Input `Y` is shape (D, N); returns `(X, Z)` with same shape, Float32.
"""
function generate_xz(Y::AbstractMatrix, σ::Real)
    D, N = size(Y)
    T = Float32
    Z = randn(T, D, N)
    X = @views T.(Y) .+ T(σ) .* Z
    return X, Z
end

"""
    generate_data(obs, σ::Real)

Generate inputs and targets for fixed noise level: X = Y + σ·ε, Z = ε.
`obs` is (D, N). Returns `(X, Z)` as Float32 matrices (D, N).
"""
function generate_data(obs, σ::Real)
    return generate_xz(obs, σ)
end


############################
#  Model & losses          #
############################

"""
    swish(x)

Swish activation: x .* σ(x)
"""
swish(x) = x .* Flux.sigmoid.(x)

"""
    create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)

Create an MLP with architecture defined by `neurons`.
"""
function create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)
    layers = Vector{Any}(undef, length(neurons) - 1)
    for i in 1:length(neurons)-2
        layers[i] = Flux.Dense(neurons[i], neurons[i+1], activation)
    end
    layers[end] = Flux.Dense(neurons[end-1], neurons[end], last_activation)
    return Flux.Chain(layers...)
end

"""
    loss_score(nn, inputs, targets)

Compute mean squared error between predictions and targets.
"""
function loss_score(nn, inputs, targets)
    Flux.mse(nn(inputs), targets)
end

"""
    weighted_mse(Ŷ, Y, w)

Weighted MSE across batch columns using non-negative weights `w` (length B).
Returns sum_j w_j * ||Ŷ[:,j] - Y[:,j]||^2 / (sum(w) * D).
"""
function weighted_mse(Ŷ::AbstractMatrix, Y::AbstractMatrix, w::AbstractVector)
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
"""
function train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ::Real; 
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
               use_gpu=true, verbose=false)

    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)

    losses = Float32[]
    epoch_iterator = verbose ? ProgressBar(1:n_epochs) : (1:n_epochs)
    for _ in epoch_iterator
        X, Z = generate_data(obs, σ)
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
    end

    return nn, losses
end

############################
#  Divergence from score   #
############################

"""
    divergence_hutchinson(nn, X; probes=1, rademacher=true)

Estimate ∇·f(X) for f = nn (ε-predictor) via Hutchinson's estimator.
Input `X` is (D, B). Returns a (1, B) array with divergence per column.
Works on CPU/GPU depending on `X` device.
"""
function divergence_hutchinson(nn::Flux.Chain, X::AbstractMatrix;
                               probes::Integer=1, rademacher::Bool=true)
    T = eltype(X)
    B = size(X, 2)
    div = similar(X, T, 1, B); fill!(div, zero(T))
    for _ in 1:probes
        # Allocate V to match nn(X) output shape and device
        Y = nn(X)
        V = similar(Y)
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
    divergence_from_eps(nn, X; σ, convention=:unit, probes=1, rademacher=true)

Compute ∇·sθ(X) when the score is parameterized via ε̂(X):
  sθ(X) = -(1/c) ε̂(X),  with  c = 2σ for `:unit`,  c = σ for `:half`.
Thus ∇·sθ = -(1/c) ∇·ε̂.
"""
function divergence_from_eps(nn::Flux.Chain, X::AbstractMatrix;
                             σ::Real, convention::Symbol=:unit,
                             probes::Integer=1, rademacher::Bool=true)
    c = (convention === :unit) ? (2f0 * Float32(σ)) : (1f0 * Float32(σ))
    d_eps = divergence_hutchinson(nn, X; probes=probes, rademacher=rademacher)
    return @. -(d_eps) / c
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

    epoch_iterator = verbose ? ProgressBar(1:n_epochs) : (1:n_epochs)
    for _ in epoch_iterator
        X, Z = generate_data(obs, σ)
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
    end
    return nn, losses
end

"""
    train(obs_tuple, n_epochs, batch_size, neurons; kwargs...)

Train on a precomputed dataset `(X, Z)` (e.g., from KGMM centers).
"""
function train(obs_tuple, n_epochs, batch_size, neurons; 
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
               use_gpu=true, verbose=false)

    X, Z = obs_tuple
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    if verbose
        println("Using " * (device === gpu ? "GPU" : "CPU"))
    end

    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    epoch_iterator = verbose ? ProgressBar(1:n_epochs) : (1:n_epochs)
    for _ in epoch_iterator
        ep = zero(Float32)
        for (Xb, Zb) in loader
            loss, grads = Flux.withgradient(nn) do m
                Flux.mse(m(Xb), Zb)
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
    end

    return nn, losses
end

"""
    train((X, Z, w), n_epochs, batch_size, neurons; ...)

Train on a precomputed weighted dataset where `w` are per-sample weights
(e.g., KGMM cluster counts). Uses weighted MSE.
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

    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device |> Flux.f32
    opt_state = Flux.setup(opt, nn)
    Xd, Zd = device(Float32.(X)), device(Float32.(Z))
    wd = device(Float32.(w))
    _ = nn(@view Xd[:, 1:min(batch_size, size(Xd,2))])
    loader = Flux.DataLoader((Xd, Zd, wd), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    epoch_iterator = verbose ? ProgressBar(1:n_epochs) : (1:n_epochs)
    for _ in epoch_iterator
        ep = zero(Float32)
        for (Xb, Zb, wb) in loader
            loss, grads = Flux.withgradient(nn) do m
                weighted_mse(m(Xb), Zb, vec(wb))
            end
            Flux.update!(opt_state, nn, grads[1])
            ep += Float32(loss)
        end
        push!(losses, ep / length(loader))
    end

    return nn, losses
end

"""
    check_loss(obs, nn, σ::Real; n_samples=1, verbose=false)

Evaluate average MSE over random draws X = Y + σ·ε.
"""
function check_loss(obs, nn, σ::Real; n_samples=1, verbose=false)
    loss = 0.0
    sample_iterator = verbose ? ProgressBar(1:n_samples) : (1:n_samples)
    for _ in sample_iterator
        X, Z = generate_data(obs, σ)
        loss += loss_score(nn, X, Z)
    end
    return loss / n_samples
end


############################
#  KGMM integration        #
############################

"""
    dataset_from_kgmm(res::NamedTuple, σ::Real; convention=:unit)

Build a supervised (X, Z) dataset from KGMM result `res`.
For convention=:unit (default in KGMM), s = -E[z]/(2σ) ⇒ Z = -2σ·s.
For convention=:half, s = -E[z]/σ ⇒ Z = -σ·s.
"""
function dataset_from_kgmm(res::NamedTuple, σ::Real; convention::Symbol=:unit)
    X = Float32.(res.centers)
    S = Float32.(res.score)
    scale = convention === :unit ? -2f0 * Float32(σ) : -1f0 * Float32(σ)
    Z = @. scale * S
    return X, Z
end

"""
    train(obs; preprocessing=false, σ=0.1, neurons=[D,128,128,D],
          n_epochs=50, batch_size=8192, lr=1e-2, use_gpu=false, verbose=true,
          kgmm_kwargs=(;), kgmm_convention=:unit)

High-level wrapper:
- preprocessing=false: ε-training from random draws X=Y+σ·ε.
- preprocessing=true : run KGMM(σ, obs; kgmm_kwargs...) and supervise on centroids.
Returns `(nn, train_losses, val_losses, div_fn, kgmm)`; `val_losses` is empty in this wrapper.
`kgmm` is the output of KGMM(σ, obs; ...) when `preprocessing=true`, otherwise `nothing`.
"""
function train(obs::AbstractMatrix; preprocessing::Bool=false,
               σ::Real=0.1,
               neurons::Vector{Int}=[size(obs,1),128,128,size(obs,1)],
               n_epochs::Int=50,
               batch_size::Int=8192,
               lr::Float64=1e-2,
               use_gpu::Bool=false,
               verbose::Bool=true,
               kgmm_kwargs=(;),
               kgmm_convention::Symbol=:unit,
               divergence::Bool=false,
               probes::Integer=1,
               rademacher::Bool=true)

    opt = Flux.Adam(Float32(lr))
    device = (use_gpu && CUDA.functional()) ? gpu : cpu

    if !preprocessing
        nn, losses = train(obs, n_epochs, batch_size, neurons, σ; opt=opt, use_gpu=use_gpu, verbose=verbose)
        div_fn = if divergence
            let nn=nn, device=device, probes=probes, rademacher=rademacher, σ=σ, conv=kgmm_convention
                X -> begin
                    Xd = device(Float32.(X))
                    Array(divergence_from_eps(nn, Xd; σ=σ, convention=conv, probes=probes, rademacher=rademacher))
                end
            end
        else
            X -> zeros(eltype(X), 1, size(X,2))
        end
        return nn, losses, Float32[], div_fn, nothing
    else
        # Compute KGMM via the project module; expects `ScoreEstimation` to be in scope.
        res = ScoreEstimation.KGMM(σ, obs; kgmm_kwargs...)
        X, Z = dataset_from_kgmm(res, σ; convention=kgmm_convention)
        w    = Float32.(res.counts)
        nn, losses = train((X, Z, w), n_epochs, batch_size, neurons; opt=opt, use_gpu=use_gpu, verbose=verbose)
        div_fn = if divergence
            let nn=nn, device=device, probes=probes, rademacher=rademacher, σ=σ, conv=kgmm_convention
                Xq -> begin
                    Xd = device(Float32.(Xq))
                    Array(divergence_from_eps(nn, Xd; σ=σ, convention=conv, probes=probes, rademacher=rademacher))
                end
            end
        else
            Xq -> zeros(eltype(Xq), 1, size(Xq,2))
        end
        return nn, losses, Float32[], div_fn, res
    end
end
