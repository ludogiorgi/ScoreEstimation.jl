function sparse_generator(labels::AbstractVector{<:Integer},
                          n_states::Int;
                          dt::Real,
                          finite_dt_correction::Bool=false,
                          finite_dt_correction_mode::Symbol=:per_state)
    nsteps = length(labels)
    nsteps >= 2 || error("At least two labels are required to estimate a generator.")

    transition_counts = Dict{Tuple{Int,Int},Int}()
    state_out = zeros(Int, n_states)
    state_leaves = zeros(Int, n_states)

    @inbounds for t in 1:(nsteps - 1)
        i = labels[t]
        j = labels[t + 1]
        state_out[i] += 1
        if i != j
            state_leaves[i] += 1
            key = (i, j)
            transition_counts[key] = get(transition_counts, key, 0) + 1
        end
    end

    leave_fraction = sum(values(transition_counts)) / max(nsteps - 1, 1)
    if leave_fraction > 0.3
        correction = -log1p(-clamp(leave_fraction, 0.0, 1.0 - eps(Float64))) / max(leave_fraction, eps(Float64))
        if !finite_dt_correction
            @warn "High cluster-switch probability per dt; enable finite-Δt correction or coarsen the partition." dt leave_fraction correction
        else
            @info "Applying finite-Δt correction to the coarse generator" dt leave_fraction mode=finite_dt_correction_mode correction
        end
    end

    col_scale = ones(Float64, n_states)
    if finite_dt_correction
        finite_dt_correction_mode in (:per_state, :mean_diag_scale) ||
            error("finite_dt_correction_mode must be :per_state or :mean_diag_scale.")

        if finite_dt_correction_mode === :per_state
            @inbounds for i in 1:n_states
                n_out = state_out[i]
                n_out == 0 && continue
                p_leave = clamp(state_leaves[i] / n_out, eps(Float64), 1.0)
                p_stay = clamp(1.0 - p_leave, eps(Float64), 1.0)
                col_scale[i] = -log(p_stay) / p_leave
            end
        else
            alpha_naive = 0.0
            alpha_corr = 0.0
            contributing = 0
            @inbounds for i in 1:n_states
                n_out = state_out[i]
                n_out == 0 && continue
                p_leave = clamp(state_leaves[i] / n_out, 0.0, 1.0)
                p_stay = clamp(1.0 - p_leave, eps(Float64), 1.0)
                alpha_naive += -(p_leave / dt)
                alpha_corr += log(p_stay) / dt
                contributing += 1
            end
            contributing > 0 || error("Unable to compute finite-Δt correction: no populated states.")
            alpha_naive /= contributing
            alpha_corr /= contributing
            abs(alpha_naive) > 0 || error("Unable to compute finite-Δt correction: mean naive diagonal is zero.")
            fill!(col_scale, alpha_corr / alpha_naive)
        end
    end

    nnz_estimate = length(transition_counts) + n_states
    rows = Vector{Int}(undef, nnz_estimate)
    cols = Vector{Int}(undef, nnz_estimate)
    vals = Vector{Float64}(undef, nnz_estimate)
    column_sums = zeros(Float64, n_states)

    idx = 1
    for ((i, j), count) in transition_counts
        denom = state_out[i] * dt
        rate = denom > 0 ? (col_scale[i] * count) / denom : 0.0
        rows[idx] = j
        cols[idx] = i
        vals[idx] = rate
        column_sums[i] += rate
        idx += 1
    end

    @inbounds for i in 1:n_states
        rows[idx] = i
        cols[idx] = i
        vals[idx] = -column_sums[i]
        idx += 1
    end

    return sparse(rows, cols, vals, n_states, n_states)
end

function compute_cluster_centers_and_pi(samples::AbstractMatrix,
                                        labels::AbstractVector{<:Integer},
                                        n_states::Int)
    dim, nsamples = size(samples)
    length(labels) == nsamples || error("Number of labels must match the number of samples.")

    centers = zeros(Float64, dim, n_states)
    counts = zeros(Int, n_states)

    @inbounds for idx in 1:nsamples
        label = labels[idx]
        @views centers[:, label] .+= samples[:, idx]
        counts[label] += 1
    end

    @inbounds for state in 1:n_states
        counts[state] > 0 && (@views centers[:, state] ./= counts[state])
    end

    pi_vec = counts ./ sum(counts)
    return centers, pi_vec, counts
end

function compute_generator_cdot(centers::AbstractMatrix,
                                Q::AbstractMatrix,
                                pi_vec::AbstractVector)
    pi_norm = pi_vec ./ sum(pi_vec)
    return (centers * Q * Diagonal(pi_norm)) * transpose(centers)
end

function generator_correlation(Q::AbstractMatrix,
                               pi_vec::AbstractVector,
                               observable::AbstractVector;
                               dt::Real,
                               max_lag::Int,
                               normalize::Bool=true)
    max_lag >= 0 || error("max_lag must be non-negative.")

    pi_norm = pi_vec ./ sum(pi_vec)
    centered = Float64.(observable) .- dot(observable, pi_norm)
    initial = Diagonal(pi_norm) * centered
    values = zeros(Float64, max_lag + 1)
    values[1] = dot(centered, initial)

    if max_lag == 0
        return normalize && values[1] != 0.0 ? values ./ values[1] : values
    end

    propagator = exp(dt * Matrix(Q))
    state = Vector{Float64}(initial)
    for lag in 1:max_lag
        state = propagator * state
        values[lag + 1] = dot(centered, state)
    end

    return normalize && values[1] != 0.0 ? values ./ values[1] : values
end

function compute_score_position_matrix(samples::AbstractMatrix,
                                       score_model;
                                       batch_size::Int=4096,
                                       use_gpu::Bool=false)
    dim, nsamples = size(samples)
    nsamples > 0 || error("At least one sample is required to estimate the Stein matrix.")

    model_eval = if use_gpu && _cuda_functional()
        cuda = _cuda_module()
        wrapper = _move_to_gpu(deepcopy(score_model), cuda)
        Flux.testmode!(wrapper)
        wrapper
    else
        Flux.testmode!(score_model)
        score_model
    end

    V = zeros(Float64, dim, dim)
    total = 0

    for start_idx in 1:batch_size:nsamples
        stop_idx = min(start_idx + batch_size - 1, nsamples)
        batch = Float32.(samples[:, start_idx:stop_idx])
        eval_batch = _to_device(batch; use_gpu=use_gpu)
        scores = model_eval(eval_batch)
        score_host = Float64.(Array(scores))
        batch_host = Float64.(batch)
        V .+= score_host * transpose(batch_host)
        total += size(batch_host, 2)
    end

    total > 0 || error("No samples were processed while estimating the Stein matrix.")
    return V ./ total
end

function estimate_phi_sigma(correlated_samples::AbstractMatrix,
                            score_model;
                            dt::Real,
                            stationary_samples::Union{Nothing,AbstractMatrix}=nothing,
                            perturb_scale::Union{Nothing,Real}=nothing,
                            generator_perturb_scale::Real=0.0,
                            q_min_prob::Real=1.0e-4,
                            regularization::Real=1.0e-8,
                            batch_size::Int=4096,
                            use_gpu::Bool=false,
                            use_stein_correction::Bool=true,
                            enforce_psd_symmetric::Bool=true,
                            finite_dt_correction::Bool=false,
                            finite_dt_correction_mode::Symbol=:per_state,
                            rng::AbstractRNG=Random.default_rng())
    dim, ntime = size(correlated_samples)
    ntime >= 2 || error("At least two correlated samples are required.")
    state_samples = stationary_samples === nothing ? correlated_samples : stationary_samples
    size(state_samples, 1) == dim || error("correlated_samples and stationary_samples must have the same state dimension.")

    σ = perturb_scale === nothing ? Float64(score_model.noise_scale) : Float64(perturb_scale)
    generator_σ = Float64(generator_perturb_scale)

    correlated_noisy = Float32.(correlated_samples .+ generator_σ .* randn(rng, Float32, size(correlated_samples)))
    labels = StateSpacePartition(correlated_noisy; method=Tree(false, q_min_prob)).partitions
    n_states = maximum(labels)
    Q = sparse_generator(labels, n_states;
        dt=dt,
        finite_dt_correction=finite_dt_correction,
        finite_dt_correction_mode=finite_dt_correction_mode,
    )

    centers, pi_vec, counts = compute_cluster_centers_and_pi(correlated_noisy, labels, n_states)
    cdot = compute_generator_cdot(centers, Q, pi_vec)

    stationary_noisy = Float32.(state_samples .+ σ .* randn(rng, Float32, size(state_samples)))
    V = compute_score_position_matrix(stationary_noisy, score_model; batch_size=batch_size, use_gpu=use_gpu)
    Phi_raw = use_stein_correction ? (cdot / V) : (-cdot)

    Phi_symmetric_raw = 0.5 .* (Phi_raw + transpose(Phi_raw))
    Phi_antisymmetric = 0.5 .* (Phi_raw - transpose(Phi_raw))
    eigen_symmetric = eigen(Symmetric(Phi_symmetric_raw))
    eigvals_symmetric = eigen_symmetric.values
    eigvecs_symmetric = eigen_symmetric.vectors
    clipped_eigvals = if enforce_psd_symmetric
        max.(eigvals_symmetric, regularization)
    else
        copy(eigvals_symmetric)
    end
    shift = maximum(clipped_eigvals .- eigvals_symmetric)
    Phi_symmetric = eigvecs_symmetric * Diagonal(clipped_eigvals) * transpose(eigvecs_symmetric)
    Phi = Phi_symmetric + Phi_antisymmetric
    Sigma = Matrix(cholesky(Symmetric(Phi_symmetric)).L)

    info = Dict{Symbol,Any}(
        :Q => Q,
        :centers => centers,
        :pi_vec => pi_vec,
        :counts => counts,
        :cdot => cdot,
        :V => V,
        :Phi_raw => Phi_raw,
        :Phi_symmetric_raw => Phi_symmetric_raw,
        :Phi_symmetric => Phi_symmetric,
        :Phi_antisymmetric => Phi_antisymmetric,
        :regularization_shift => shift,
        :n_states => n_states,
        :dt => Float64(dt),
        :perturb_scale => σ,
        :generator_perturb_scale => generator_σ,
        :use_stein_correction => use_stein_correction,
        :enforce_psd_symmetric => enforce_psd_symmetric,
        :finite_dt_correction => finite_dt_correction,
        :finite_dt_correction_mode => finite_dt_correction_mode,
    )
    return Phi, Sigma, info
end
