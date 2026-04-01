using ProgressMeter

struct ScoreModel{M,T,V<:AbstractVector{T}}
    model::M
    noise_scale::T
    mean::V
    scale::V
end

Flux.@functor ScoreModel

struct LinearScoreSDE{S,MP,MS}
    score_model::S
    phi::MP
    sigma::MS
end

Flux.@functor LinearScoreSDE

function ScoreModel(model, noise_scale::Real;
                    mean::AbstractVector{<:Real},
                    scale::AbstractVector{<:Real})
    mean32 = Float32.(collect(mean))
    scale32 = Float32.(collect(scale))
    return ScoreModel(model, Float32(noise_scale), mean32, scale32)
end

function LinearScoreSDE(score_model, phi::AbstractMatrix, sigma::AbstractMatrix)
    phi32 = Float32.(phi)
    sigma32 = Float32.(sigma)
    return LinearScoreSDE{typeof(score_model),typeof(phi32),typeof(sigma32)}(score_model, phi32, sigma32)
end

function _normalization_vector(values::AbstractVector, x::AbstractMatrix, ::Type{T}) where {T}
    data = convert.(T, values)
    if !(x isa Array) && _cuda_functional()
        return _to_device(data; use_gpu=true)
    end
    return data
end

function (wrapper::ScoreModel)(x::AbstractMatrix)
    T = eltype(x)
    μ = reshape(_normalization_vector(wrapper.mean, x, T), :, 1)
    σx = reshape(_normalization_vector(wrapper.scale, x, T), :, 1)
    y = (x .- μ) ./ σx
    eps_hat = wrapper.model(y)
    return (-eps_hat ./ convert(T, wrapper.noise_scale)) ./ σx
end

function (wrapper::LinearScoreSDE)(x::AbstractMatrix)
    return wrapper.phi * wrapper.score_model(x)
end

struct LangevinWorkCache{TX}
    drift::TX
    noise::TX
end

struct AffineLangevinWorkCache{TX}
    drift::TX
    noise::TX
    mixed_noise::TX
end

struct CPUScoreState{TX,TM,TC}
    x::TX
    model::TM
    cache::TC
end

struct GPUScoreState{TX,TM,TC}
    x::TX
    model::TM
    cache::TC
end

const CPU_SCORE_STATE_CACHE = IdDict{Any,Any}()
const GPU_SCORE_STATE_CACHE = IdDict{Any,Any}()
const MAX_PROGRESS_UPDATES = 200
const PROGRESS_MIN_INTERVAL = 0.5

function _seed_rng!(device::Symbol, seed::Union{Nothing,Integer})
    seed === nothing && return nothing
    Random.seed!(seed)
    if device === :gpu
        cuda = _cuda_module()
        cuda === nothing && return nothing
        Base.invokelatest(getproperty(cuda, Symbol("seed!")), seed)
    end
    return nothing
end

function _init_progress(total_steps::Int;
                        progress::Union{Bool,ProgressMeter.Progress,Nothing},
                        desc::AbstractString)
    if total_steps <= 0 || progress === false || progress === nothing
        return nothing, 0, false
    end

    stride = max(1, fld(total_steps, MAX_PROGRESS_UPDATES))
    if progress isa ProgressMeter.Progress
        return progress, stride, false
    end
    return Progress(total_steps; desc=desc, dt=PROGRESS_MIN_INTERVAL), stride, true
end

@inline function _tick_progress!(progress, step::Int, stride::Int, total_steps::Int)
    if progress !== nothing && (step % stride == 0 || step == total_steps)
        ProgressMeter.update!(progress, step)
    end
    return nothing
end

@inline function _finish_progress!(progress, owns_progress::Bool)
    owns_progress && progress !== nothing && ProgressMeter.finish!(progress)
    return nothing
end

function _setup_cache(x::AbstractMatrix, model)
    drift = similar(model(x))
    noise = similar(x)
    return LangevinWorkCache(drift, noise)
end

function _setup_cache(x::AbstractMatrix, model::LinearScoreSDE)
    drift = similar(model(x))
    noise = similar(x)
    mixed_noise = similar(x)
    return AffineLangevinWorkCache(drift, noise, mixed_noise)
end

_move_to_gpu(x, cuda) = Base.invokelatest(cuda.cu, x)

function _move_to_gpu(layer::Flux.Dense, cuda)
    weight = Base.invokelatest(cuda.cu, layer.weight)
    bias = Base.invokelatest(cuda.cu, layer.bias)
    return Flux.Dense(weight, bias, layer.σ)
end

function _move_to_gpu(chain::Flux.Chain, cuda)
    return Flux.Chain((_move_to_gpu(layer, cuda) for layer in chain.layers)...)
end

function _move_to_gpu(wrapper::ScoreModel, cuda)
    return ScoreModel(
        _move_to_gpu(wrapper.model, cuda),
        wrapper.noise_scale;
        mean=Base.invokelatest(cuda.cu, wrapper.mean),
        scale=Base.invokelatest(cuda.cu, wrapper.scale),
    )
end

function _move_to_gpu(wrapper::LinearScoreSDE, cuda)
    score_gpu = _move_to_gpu(wrapper.score_model, cuda)
    phi_gpu = Base.invokelatest(cuda.cu, wrapper.phi)
    sigma_gpu = Base.invokelatest(cuda.cu, wrapper.sigma)
    return LinearScoreSDE{typeof(score_gpu),typeof(phi_gpu),typeof(sigma_gpu)}(
        score_gpu,
        phi_gpu,
        sigma_gpu,
    )
end

function _cpu_state(score_model, x0::AbstractMatrix)
    return get!(CPU_SCORE_STATE_CACHE, score_model) do
        model_cpu = deepcopy(score_model)
        Flux.testmode!(model_cpu)
        x = Array{Float32}(undef, size(x0)...)
        copyto!(x, Float32.(x0))
        cache = _setup_cache(x, model_cpu)
        CPUScoreState(x, model_cpu, cache)
    end
end

function _gpu_state(score_model, x0::AbstractMatrix)
    cuda = _cuda_module()
    cuda === nothing && error("CUDA is unavailable")
    return get!(GPU_SCORE_STATE_CACHE, score_model) do
        model_gpu = _move_to_gpu(deepcopy(score_model), cuda)
        Flux.testmode!(model_gpu)
        x = Base.invokelatest(cuda.cu, Float32.(x0))
        cache = _setup_cache(x, model_gpu)
        GPUScoreState(x, model_gpu, cache)
    end
end

function _resolve_device(device)
    requested = device isa Symbol ? device : Symbol(device)
    if requested === :auto
        return _cuda_functional() ? :gpu : :cpu
    end
    if requested === :gpu && !_cuda_functional()
        @warn "GPU requested but CUDA.functional() is false. Falling back to CPU."
        return :cpu
    end
    return requested
end

function _enforce_boundary!(x::AbstractMatrix, x0::AbstractMatrix, boundary::Tuple{<:Real,<:Real})
    lo, hi = boundary
    dim, nens = size(x)
    @inbounds for j in 1:nens
        violated = false
        for i in 1:dim
            val = x[i, j]
            if val < lo || val > hi
                violated = true
                break
            end
        end
        if violated
            @views x[:, j] .= x0[:, j]
        end
    end
    return nothing
end

function _enforce_boundary_gpu!(x, x0, boundary::Tuple{<:Real,<:Real})
    T = eltype(x)
    lo, hi = boundary
    violated = vec(reduce(|, (x .< T(lo)) .| (x .> T(hi)); dims=1))
    violated_host = Array(violated)
    @inbounds for (j, do_reset) in pairs(violated_host)
        do_reset || continue
        @views x[:, j] .= x0[:, j]
    end
    return nothing
end

function _em_step!(x, cache::LangevinWorkCache, model, dt::Real, sqrt_2dt::Real)
    copyto!(cache.drift, model(x))
    randn!(cache.noise)
    T = eltype(x)
    dtT = T(dt)
    sqrt_2dtT = T(sqrt_2dt)
    @. x += dtT * cache.drift + sqrt_2dtT * cache.noise
    return nothing
end

function _em_step!(x, cache::AffineLangevinWorkCache, model::LinearScoreSDE, dt::Real, sqrt_2dt::Real)
    copyto!(cache.drift, model(x))
    randn!(cache.noise)
    mul!(cache.mixed_noise, model.sigma, cache.noise)
    T = eltype(x)
    dtT = T(dt)
    sqrt_2dtT = T(sqrt_2dt)
    @. x += dtT * cache.drift + sqrt_2dtT * cache.mixed_noise
    return nothing
end

function evolve_score_langevin(score_model,
                               x0::AbstractMatrix;
                               dt::Real,
                               n_steps::Integer,
                               device::Union{Symbol,String}=:auto,
                               seed::Union{Nothing,Integer}=nothing,
                               boundary=nothing,
                               progress::Union{Bool,ProgressMeter.Progress,Nothing}=false,
                               progress_desc::AbstractString="Score Langevin integration")
    device_sym = _resolve_device(device)
    _seed_rng!(device_sym, seed)
    sqrt_2dt = sqrt(2 * dt)
    n_steps_int = max(Int(n_steps), 1)

    if device_sym === :gpu
        cuda = _cuda_module()
        state = _gpu_state(score_model, x0)
        copyto!(state.x, Base.invokelatest(cuda.cu, Float32.(x0)))
        x0_dev = boundary === nothing ? nothing : copy(state.x)
        progress_meter, stride, owns_progress = _init_progress(n_steps_int; progress=progress, desc=progress_desc)
        for step in 1:n_steps_int
            _em_step!(state.x, state.cache, state.model, dt, sqrt_2dt)
            boundary === nothing || _enforce_boundary_gpu!(state.x, x0_dev, boundary)
            _tick_progress!(progress_meter, step, stride, n_steps_int)
        end
        _finish_progress!(progress_meter, owns_progress)
        return Array(state.x)
    end

    state = _cpu_state(score_model, x0)
    copyto!(state.x, Float32.(x0))
    x0_cpu = boundary === nothing ? nothing : copy(state.x)
    progress_meter, stride, owns_progress = _init_progress(n_steps_int; progress=progress, desc=progress_desc)
    for step in 1:n_steps_int
        _em_step!(state.x, state.cache, state.model, dt, sqrt_2dt)
        boundary === nothing || _enforce_boundary!(state.x, x0_cpu, boundary)
        _tick_progress!(progress_meter, step, stride, n_steps_int)
    end
    _finish_progress!(progress_meter, owns_progress)
    return copy(state.x)
end

function evolve_score_langevin_snapshots(score_model,
                                         x0::AbstractMatrix;
                                         dt::Real,
                                         n_steps::Integer,
                                         burn_in::Integer=0,
                                         resolution::Integer=1,
                                         device::Union{Symbol,String}=:auto,
                                         seed::Union{Nothing,Integer}=nothing,
                                         boundary=nothing,
                                         progress::Union{Bool,ProgressMeter.Progress,Nothing}=false,
                                         progress_desc::AbstractString="Score Langevin snapshots")
    total_steps = max(Int(n_steps), 1)
    burn_in_steps = max(Int(burn_in), 0)
    steps_per_snapshot = max(Int(resolution), 1)
    total_snapshots = fld(total_steps, steps_per_snapshot)
    burn_in_snapshots = fld(burn_in_steps, steps_per_snapshot)
    kept_snapshots = total_snapshots - burn_in_snapshots
    kept_snapshots > 0 || error("Burn-in removes all snapshots. Increase n_steps or reduce burn_in/resolution.")

    dim, nens = size(x0)
    traj = Array{Float32}(undef, dim, kept_snapshots, nens)
    device_sym = _resolve_device(device)
    _seed_rng!(device_sym, seed)
    sqrt_2dt = sqrt(2 * dt)

    if device_sym === :gpu
        cuda = _cuda_module()
        state = _gpu_state(score_model, x0)
        copyto!(state.x, Base.invokelatest(cuda.cu, Float32.(x0)))
        x0_dev = boundary === nothing ? nothing : copy(state.x)
        snapshot_idx = 0
        progress_meter, stride, owns_progress = _init_progress(total_steps; progress=progress, desc=progress_desc)
        for step in 1:total_steps
            _em_step!(state.x, state.cache, state.model, dt, sqrt_2dt)
            boundary === nothing || _enforce_boundary_gpu!(state.x, x0_dev, boundary)
            if step > burn_in_steps && step % steps_per_snapshot == 0
                snapshot_idx += 1
                @views traj[:, snapshot_idx, :] .= Array(state.x)
            end
            _tick_progress!(progress_meter, step, stride, total_steps)
        end
        _finish_progress!(progress_meter, owns_progress)
        return traj
    end

    state = _cpu_state(score_model, x0)
    copyto!(state.x, Float32.(x0))
    x0_cpu = boundary === nothing ? nothing : copy(state.x)
    snapshot_idx = 0
    progress_meter, stride, owns_progress = _init_progress(total_steps; progress=progress, desc=progress_desc)
    for step in 1:total_steps
        _em_step!(state.x, state.cache, state.model, dt, sqrt_2dt)
        boundary === nothing || _enforce_boundary!(state.x, x0_cpu, boundary)
        if step > burn_in_steps && step % steps_per_snapshot == 0
            snapshot_idx += 1
            @views traj[:, snapshot_idx, :] .= state.x
        end
        _tick_progress!(progress_meter, step, stride, total_steps)
    end
    _finish_progress!(progress_meter, owns_progress)
    return traj
end

function evolve_affine_score_langevin(score_model,
                                      x0::AbstractMatrix,
                                      phi::AbstractMatrix,
                                      sigma::AbstractMatrix;
                                      kwargs...)
    return evolve_score_langevin(LinearScoreSDE(score_model, phi, sigma), x0; kwargs...)
end

function evolve_affine_score_langevin_snapshots(score_model,
                                                x0::AbstractMatrix,
                                                phi::AbstractMatrix,
                                                sigma::AbstractMatrix;
                                                kwargs...)
    return evolve_score_langevin_snapshots(LinearScoreSDE(score_model, phi, sigma), x0; kwargs...)
end

@inline function _find_cell(grid::AbstractVector{<:Real}, x::Real)
    x <= grid[1] && return 1, 1, 0.0f0
    x >= grid[end] && return length(grid), length(grid), 0.0f0
    hi = searchsortedfirst(grid, x)
    lo = hi - 1
    denom = grid[hi] - grid[lo]
    w = denom <= 0 ? 0.0f0 : Float32((x - grid[lo]) / denom)
    return lo, hi, w
end

function _interp_vector_2d!(out::AbstractVector,
                            grid::AbstractArray{<:Real,3},
                            xgrid::AbstractVector{<:Real},
                            ygrid::AbstractVector{<:Real},
                            x::Real,
                            y::Real)
    ix0, ix1, wx = _find_cell(xgrid, x)
    iy0, iy1, wy = _find_cell(ygrid, y)
    w00 = (1.0f0 - wx) * (1.0f0 - wy)
    w10 = wx * (1.0f0 - wy)
    w01 = (1.0f0 - wx) * wy
    w11 = wx * wy

    @inbounds for i in eachindex(out)
        out[i] =
            w00 * Float32(grid[i, ix0, iy0]) +
            w10 * Float32(grid[i, ix1, iy0]) +
            w01 * Float32(grid[i, ix0, iy1]) +
            w11 * Float32(grid[i, ix1, iy1])
    end
    return nothing
end

function _interp_matrix_2d!(out::AbstractMatrix,
                            grid::AbstractArray{<:Real,4},
                            xgrid::AbstractVector{<:Real},
                            ygrid::AbstractVector{<:Real},
                            x::Real,
                            y::Real)
    ix0, ix1, wx = _find_cell(xgrid, x)
    iy0, iy1, wy = _find_cell(ygrid, y)
    w00 = (1.0f0 - wx) * (1.0f0 - wy)
    w10 = wx * (1.0f0 - wy)
    w01 = (1.0f0 - wx) * wy
    w11 = wx * wy

    @inbounds for j in axes(out, 2), i in axes(out, 1)
        out[i, j] =
            w00 * Float32(grid[i, j, ix0, iy0]) +
            w10 * Float32(grid[i, j, ix1, iy0]) +
            w01 * Float32(grid[i, j, ix0, iy1]) +
            w11 * Float32(grid[i, j, ix1, iy1])
    end
    return nothing
end

function evolve_interpolated_langevin_snapshots(x0::AbstractMatrix,
                                                xgrid::AbstractVector{<:Real},
                                                ygrid::AbstractVector{<:Real},
                                                drift_grid::AbstractArray{<:Real,3},
                                                sigma_grid::AbstractArray{<:Real,4};
                                                dt::Real,
                                                n_steps::Integer,
                                                burn_in::Integer=0,
                                                resolution::Integer=1,
                                                seed::Union{Nothing,Integer}=nothing,
                                                boundary=nothing,
                                                progress::Union{Bool,ProgressMeter.Progress,Nothing}=false,
                                                progress_desc::AbstractString="Interpolated Langevin snapshots")
    length(xgrid) >= 2 || error("xgrid must contain at least two nodes.")
    length(ygrid) >= 2 || error("ygrid must contain at least two nodes.")
    size(drift_grid, 1) == size(x0, 1) || error("Drift grid dimension does not match state dimension.")
    size(sigma_grid, 1) == size(x0, 1) || error("Sigma grid dimension does not match state dimension.")
    size(sigma_grid, 2) == size(x0, 1) || error("Sigma grid second dimension does not match state dimension.")

    total_steps = max(Int(n_steps), 1)
    burn_in_steps = max(Int(burn_in), 0)
    steps_per_snapshot = max(Int(resolution), 1)
    total_snapshots = fld(total_steps, steps_per_snapshot)
    burn_in_snapshots = fld(burn_in_steps, steps_per_snapshot)
    kept_snapshots = total_snapshots - burn_in_snapshots
    kept_snapshots > 0 || error("Burn-in removes all snapshots. Increase n_steps or reduce burn_in/resolution.")

    _seed_rng!(:cpu, seed)
    sqrt_2dt = sqrt(2 * dt)
    dim, nens = size(x0)
    traj = Array{Float32}(undef, dim, kept_snapshots, nens)
    x = Array{Float32}(undef, dim, nens)
    copyto!(x, Float32.(x0))
    x0_cpu = boundary === nothing ? nothing : copy(x)
    drift = Vector{Float32}(undef, dim)
    ξ = Vector{Float32}(undef, dim)
    sigma = Matrix{Float32}(undef, dim, dim)
    delta = Vector{Float32}(undef, dim)
    snapshot_idx = 0

    progress_meter, stride, owns_progress = _init_progress(total_steps; progress=progress, desc=progress_desc)
    for step in 1:total_steps
        @inbounds for ens in 1:nens
            _interp_vector_2d!(drift, drift_grid, xgrid, ygrid, x[1, ens], x[2, ens])
            _interp_matrix_2d!(sigma, sigma_grid, xgrid, ygrid, x[1, ens], x[2, ens])
            randn!(ξ)
            mul!(delta, sigma, ξ)
            x[1, ens] += Float32(dt) * drift[1] + Float32(sqrt_2dt) * delta[1]
            x[2, ens] += Float32(dt) * drift[2] + Float32(sqrt_2dt) * delta[2]
        end
        boundary === nothing || _enforce_boundary!(x, x0_cpu, boundary)
        if step > burn_in_steps && step % steps_per_snapshot == 0
            snapshot_idx += 1
            @views traj[:, snapshot_idx, :] .= x
        end
        _tick_progress!(progress_meter, step, stride, total_steps)
    end
    _finish_progress!(progress_meter, owns_progress)
    return traj
end
