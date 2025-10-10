#############################
#  preprocessing.jl (kgmm)  #
#############################
# Functions only. No `module`/`using`/`export` here.
#
# This file implements a kernel GMM estimator over a state-space
# partition. The public entry point is `kgmm`, which accepts a noise
# scale σ (scalar or vector) and a data matrix μ, and returns for each σ:
#   - centers        : per-cluster centroids (E[x | cluster])
#   - score          : score at centers (standard: s = -(1/σ) E[z|x])
#   - divergence     : divergence of the score (same convention)
#   # (disabled) score_residual : score reconstructed from (Emu - centers)/σ^2
#   - counts         : sample counts per cluster (accumulated during EMA)
#   - Nc             : number of clusters
#   - partition      : the StateSpacePartition object
#
# Estimation outline (for each σ):
#   1) Build a partition once from an initial noisy draw x0 = μ + σ*z0.
#   2) Run an exponential moving average (EMA) of per-cluster means
#      across additional noisy batches until convergence.
#   3) While running EMA, also accumulate ∑‖z‖² and counts per cluster.
#   4) From the accumulated statistics, compute score and divergence.
#
# Parallelization:
#   - Threaded labeling (per column)
#   - Thread-local accumulators and single global reduction
#   - In-place buffers in hot loops
#
# Score convention (standard Gaussian smoothing with std σ):
#   s(x) = -(1/σ) E[z | x]

# --------------------------------------------------------------------------
# Internal utilities
# --------------------------------------------------------------------------

"""
    _draw_noisy!(x, z, μ, σ)

Draw `z ~ N(0, I)` in-place and set `x = μ + σ*z` in-place. Returns `(x, z)`.
The sizes of `x`, `z`, and `μ` must match.
"""
@inline function _draw_noisy!(x::AbstractMatrix, z::AbstractMatrix,
                              μ::AbstractMatrix, σ::Real)
    @assert size(x) == size(z) == size(μ)
    randn!(z)
    @inbounds @. x = μ + σ*z
    return x, z
end

"""
    _assign_labels!(labels, x, ssp)

Assign cluster labels to columns of `x` using `ssp.embedding`. The result
is written into `labels` in-place and also returned.
"""
@inline function _assign_labels!(labels::AbstractVector{Int},
                                 x::AbstractMatrix, ssp)
    @inbounds Threads.@threads for i in eachindex(labels)
        labels[i] = ssp.embedding(view(x, :, i))
    end
    return labels
end

"""
    _batch_stats(labels, z, x, μ) -> (Ez, Emu, Xc, z2_sums, cnts)

Compute per-cluster statistics over one batch:
  Ez       = E[z | cluster]
  Emu      = E[μ | cluster]
  Xc       = E[x | cluster]
  z2_sums  = ∑‖z‖² per cluster (not normalized)
  cnts     = counts per cluster

Thread-local accumulators are used, followed by a single global reduction.
"""
function _batch_stats(labels::AbstractVector{<:Integer},
                      z::AbstractMatrix{<:Real},
                      x::AbstractMatrix{<:Real},
                      μ::AbstractMatrix{<:Real})
    d, N = size(z)
    C = maximum(labels)
    T = Float64

    nt = Threads.nthreads()
    l_sum_z  = [zeros(T, d, C) for _ in 1:nt]
    # l_sum_μ  = [zeros(T, d, C) for _ in 1:nt]   # (commented out: average μ)
    l_sum_x  = [zeros(T, d, C) for _ in 1:nt]
    l_sum_z2 = [zeros(T,     C) for _ in 1:nt]
    l_cnt    = [zeros(Int,    C) for _ in 1:nt]

    @inbounds Threads.@threads for i in 1:N
        t  = Threads.threadid()
        c  = labels[i]
        vzi = view(z, :, i)
        vxi = view(x, :, i)
        # vμi = view(μ, :, i)                      # (commented)
        vSz = view(l_sum_z[t],  :, c)
        # vSμ = view(l_sum_μ[t],  :, c)            # (commented)
        vSx = view(l_sum_x[t],  :, c)
        @inbounds @simd for j in 1:d
            vSz[j] += vzi[j]
            # vSμ[j] += vμi[j]                     # (commented)
            vSx[j] += vxi[j]
        end
        l_sum_z2[t][c] += dot(vzi, vzi)
        l_cnt[t][c]    += 1
    end

    sum_z  = zeros(T, d, C)
    # sum_μ  = zeros(T, d, C)                      # (commented)
    sum_x  = zeros(T, d, C)
    sum_z2 = zeros(T,     C)
    cnt    = zeros(Int,    C)
    @inbounds for t in 1:nt
        sum_z  .+= l_sum_z[t]
        # sum_μ  .+= l_sum_μ[t]                    # (commented)
        sum_x  .+= l_sum_x[t]
        sum_z2 .+= l_sum_z2[t]
        cnt    .+= l_cnt[t]
    end

    Ez  = zeros(T, d, C)
    # Emu = zeros(T, d, C)                         # (commented)
    Xc  = zeros(T, d, C)
    @inbounds for c in 1:C
        n = cnt[c]
        if n > 0
            invn = inv(float(n))
            vEz  = view(Ez,  :, c); vXc = view(Xc, :, c)
            vSz  = view(sum_z,  :, c); # vSμ = view(sum_μ,  :, c)   # (commented)
            vSx = view(sum_x,  :, c)
            @inbounds @simd for j in 1:d
                vEz[j]  = vSz[j]  * invn
                # vEmu[j] = vSμ[j]  * invn           # (commented)
                vXc[j]  = vSx[j]  * invn
            end
        end
    end

    return Ez, Xc, sum_z2, cnt
end

"""
    _ema_partition_means_collect(σ, μ; prob=1e-3, do_print=false, conv_param=1e-2, i_max=150)

Compute per-cluster EMA means for a fixed σ and, **during EMA**, accumulate
`∑‖z‖²` and counts per cluster.

Returns `(Ez, Emu, Xc, Nc, ssp, Ezz2, counts)` where `Ezz2 = sum_z2 ./ counts`
(with `NaN` for empty clusters).
"""
function _ema_partition_means_collect(σ::Float64, μ::AbstractMatrix{<:Real};
                                      prob::Float64=0.001,
                                      do_print::Bool=false,
                                      conv_param::Float64=1e-1,
                                      i_max::Int=150)
    # Initial draw x0 = μ + σ*z0
    x0 = similar(μ, Float64)
    z0 = similar(μ, Float64)
    _draw_noisy!(x0, z0, μ, σ)

    # Partition built once from x0
    method = Tree(false, prob)
    ssp = StateSpacePartition(x0; method=method)
    C = maximum(ssp.partitions)
    do_print && println("Number of clusters: $C")

    # First-batch stats (includes z2 and counts)
    labels0 = Vector{Int}(undef, size(μ, 2))
    _assign_labels!(labels0, x0, ssp)
    Ez, Xc, z2_sums, cnts = _batch_stats(labels0, z0, x0, μ)

    # Buffers reused across iterations
    d, N = size(μ)
    x = similar(μ, Float64)
    z = similar(μ, Float64)
    labels = Vector{Int}(undef, N)

    Ez_old, Xc_old = Ez, Xc
    Ez_new  = similar(Ez_old)
    Xc_new  = similar(Xc_old)

    # Accumulators across EMA
    sum_z2 = copy(z2_sums)
    counts = copy(cnts)
    # Keep running sums of means to weight by counts per cluster across batches
    sum_z_means = Ez .* reshape(float.(counts), 1, :)
    sum_x_means = Xc .* reshape(float.(counts), 1, :)

    Δ = 1.0
    i = 1
    while Δ > conv_param && i < i_max
        _draw_noisy!(x, z, μ, σ)
        _assign_labels!(labels, x, ssp)
        Ez_b, Xc_b, z2_b, cnt_b = _batch_stats(labels, z, x, μ)

        # Update running sums for means using counts-weighted batch contributions
        @inbounds for c in 1:C
            n_new = cnt_b[c]
            if n_new > 0
                @views sum_z_means[:, c] .+= Ez_b[:, c] .* float(n_new)
                @views sum_x_means[:, c] .+= Xc_b[:, c] .* float(n_new)
            end
        end

        # Fold divergence statistics into the EMA loop
        @inbounds @simd for c in 1:C
            sum_z2[c] += z2_b[c]
            counts[c] += cnt_b[c]
        end

        # Recompute means from sums with updated counts
        @inbounds for c in 1:C
            n_tot = counts[c]
            if n_tot > 0
                invn = inv(float(n_tot))
                @views Ez_new[:, c] .= sum_z_means[:, c] .* invn
                @views Xc_new[:, c] .= sum_x_means[:, c] .* invn
            else
                @views Ez_new[:, c] .= 0
                @views Xc_new[:, c] .= 0
            end
        end

        # Δ = mean(abs2, Ez_new - Ez_old) / max(mean(abs2, Ez_new), ϵ)
        num = 0.0
        den = 0.0
        @inbounds @simd for k in eachindex(Ez_new)
            diff = @fastmath (Ez_new[k] - Ez_old[k])
            num += diff * diff
            den += Ez_new[k] * Ez_new[k]
        end
        Δ = num / max(den, 1e-300)
        do_print && println("Iteration: $i, Δ: $Δ")

        Ez_old, Ez_new   = Ez_new, Ez_old
        # Emu_old, Emu_new = Emu_new, Emu_old                         # (commented)
        Xc_old,  Xc_new  = Xc_new,  Xc_old
        i += 1
    end

    # Build Ezz2 from accumulated sums
    Ezz2 = zeros(Float64, C)
    @inbounds for c in 1:C
        Ezz2[c] = counts[c] > 0 ? sum_z2[c] / counts[c] : NaN
    end

    return Ez_old, Xc_old, C, ssp, Ezz2, counts
end

"""
    _score_and_divergence(Ez, Xc, Ezz2, σ) -> (score, divergence)

Standard identities:
- s(x) = -(1/σ) E[z|x]
- ∇·s(x) = ((E[‖z‖²|x] - d)/σ²) - ‖s(x)‖²

Note: evaluation of `score_residual = (Emu - Xc)/σ^2` is disabled.
"""
@inline function _score_and_divergence(Ez::AbstractMatrix{<:Real},
                                       Xc::AbstractMatrix{<:Real},
                                       Ezz2::AbstractVector{<:Real},
                                       σ::Real)
    d, C = size(Ez)
    invσ  = 1 / σ
    invσ2 = invσ^2

    # Score from Ez
    score = Matrix{Float64}(undef, d, C)
    @inbounds for c in 1:C
        vS = view(score, :, c)
        vE = view(Ez,    :, c)
        @inbounds @simd for j in 1:d
            vS[j] = -invσ * float(vE[j])
        end
    end

    # Score reconstructed from (Emu - Xc)/σ^2 (disabled)
    # score_residual = Matrix{Float64}(undef, d, C)
    # @inbounds for c in 1:C
    #     vSr = view(score_residual, :, c)
    #     vEm = view(Emu,  :, c)
    #     vXc = view(Xc,   :, c)
    #     @inbounds @simd for j in 1:d
    #         vSr[j] = β * (float(vEm[j]) - float(vXc[j])) * invσ2
    #     end
    # end

    # Divergence (standard identity)
    divergence = Vector{Float64}(undef, C)
    @inbounds for c in 1:C
        sc2 = dot(view(score, :, c), view(score, :, c))
        divergence[c] = (float(Ezz2[c]) - d) * invσ2 - sc2
    end

    return score, divergence
end

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

"""
    kgmm(σ::Real, μ; prob=1e-3, conv_param=1e-2, i_max=150,
         show_progress=false)

    kgmm(σs::AbstractVector{<:Real}, μ; ...)

Kernel GMM estimator on a state-space partition learned at each σ.
Divergence statistics are accumulated **during** the EMA loop.
For each σ, returns a `NamedTuple` with:
  centers, score, divergence, counts, Nc, partition.
"""
function kgmm(σ::Real, μ::AbstractMatrix{<:Real};
              prob::Float64=1e-3,
              conv_param::Float64=1e-2,
              i_max::Int=150,
              show_progress::Bool=false)
    Ez, Xc, C, ssp, Ezz2, counts =
        _ema_partition_means_collect(float(σ), μ;
                                     prob=prob, do_print=show_progress,
                                     conv_param=conv_param, i_max=i_max)

    score, divergence = _score_and_divergence(Ez, Xc, Ezz2, σ)

    return (centers=Xc, score=score, divergence=divergence,
            score_residual=nothing, counts=counts, Nc=C, partition=ssp)
end

"""
    kgmm(σs::AbstractVector{<:Real}, μ; kwargs...) -> Vector{NamedTuple}

Vectorized convenience wrapper that applies `kgmm` to each σ in `σs`.
"""
function kgmm(σs::AbstractVector{<:Real}, μ::AbstractMatrix{<:Real}; kwargs...)
    out = Vector{NamedTuple}(undef, length(σs))
    @inbounds for k in eachindex(σs)
        out[k] = kgmm(float(σs[k]), μ; kwargs...)
    end
    return out
end
