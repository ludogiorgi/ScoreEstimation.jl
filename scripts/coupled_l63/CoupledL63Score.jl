Base.@kwdef struct ScoreCandidate
    noise_scale::Float64
    score_model
    losses::Vector{Float32}
    diagnostics
    metrics
    stein_matrix::Matrix{Float64}
    stein_error::Float64
    objective::Float64
    trajectory::Matrix{Float32}
    snapshots::Array{Float32,3}
    stationary::Matrix{Float32}
end

function random_subset_columns(samples::AbstractMatrix, max_cols::Int, rng::AbstractRNG)
    size(samples, 2) <= max_cols && return Matrix{Float64}(samples)
    indices = sort(randperm(rng, size(samples, 2))[1:max_cols])
    return Matrix{Float64}(samples[:, indices])
end

function time_subsample_columns(samples::AbstractMatrix, max_cols::Int)
    stride = max(1, ceil(Int, size(samples, 2) / max_cols))
    return Matrix{Float64}(samples[:, 1:stride:end]), stride
end

function standardize_samples(samples::AbstractMatrix)
    μ = vec(mean(samples; dims=2))
    σ = vec(std(samples; dims=2))
    σ .= max.(σ, 1.0e-6)
    normalized = (samples .- μ) ./ σ
    return normalized, μ, σ
end

function sample_initial_conditions(stationary::AbstractMatrix, n_ensembles::Int, rng::AbstractRNG)
    x0 = Matrix{Float32}(undef, size(stationary, 1), n_ensembles)
    for ens in 1:n_ensembles
        idx = rand(rng, axes(stationary, 2))
        @views x0[:, ens] .= Float32.(stationary[:, idx])
    end
    return x0
end

function snapshot_resolution(observed_dt::Real, rollout_dt::Real)
    ratio = observed_dt / rollout_dt
    resolution = round(Int, ratio)
    isapprox(ratio, resolution; atol=1.0e-12, rtol=1.0e-12) || error("Observed dt must be an integer multiple of rollout dt.")
    return max(resolution, 1)
end

function build_diag_config(cfg::RomConfig)
    return ScoreEstimation.BrusselatorConfig(
        maxlag=cfg.maxlag,
        phase_points=cfg.phase_points,
        verbose=false,
    )
end

function kept_snapshot_count(n_steps::Int, burn_in::Int, resolution::Int)
    total_snapshots = fld(max(n_steps, 1), max(resolution, 1))
    burn_in_snapshots = fld(max(burn_in, 0), max(resolution, 1))
    kept = total_snapshots - burn_in_snapshots
    kept > 0 || error("Burn-in removes all snapshots.")
    return kept
end

function evenly_spaced_seed_indices(n_total::Int, n_select::Int)
    n_select = clamp(n_select, 1, n_total)
    if n_select == 1
        return [1]
    end
    return unique(clamp.(round.(Int, range(1, n_total, length=n_select)), 1, n_total))
end

function build_observed_stationary_reference(data::SimulationData,
                                            params::ModelParameters,
                                            high_cfg::HighResolutionConfig,
                                            rom_cfg::RomConfig)
    observed = observed_matrix(data)
    resolution = snapshot_resolution(data.sample_dt, rom_cfg.candidate_rollout_dt)
    target_samples = kept_snapshot_count(
        rom_cfg.candidate_rollout_steps,
        rom_cfg.candidate_rollout_burnin,
        resolution,
    ) * rom_cfg.candidate_rollout_ensembles
    size(observed, 2) >= target_samples && return observed[:, 1:target_samples]

    seeds = state_matrix(data)
    seed_indices = evenly_spaced_seed_indices(size(seeds, 2), rom_cfg.candidate_rollout_ensembles)
    seed_states = seeds[:, seed_indices]
    nseed = size(seed_states, 2)
    samples_per_seed = ceil(Int, target_samples / nseed)
    total_samples = nseed * samples_per_seed
    stationary = Matrix{Float64}(undef, 2, total_samples)
    report_stride = max(cld(nseed, 10), 1)
    completed_seeds = Threads.Atomic{Int}(0)
    log_message(
        "fit_score",
        @sprintf(
            "Building stationary reference from dynamics: target_samples=%d, seeds=%d, samples_per_seed=%d",
            target_samples,
            nseed,
            samples_per_seed,
        ),
    )

    @threads for seed_id in 1:nseed
        u = Vector{Float64}(seed_states[:, seed_id])
        work = RK4Workspace(5)
        local_maxabs = maximum(abs, u)
        start_idx = (seed_id - 1) * samples_per_seed
        for sample_id in 1:samples_per_seed
            idx = start_idx + sample_id
            stationary[1, idx] = u[1]
            stationary[2, idx] = u[2]
            if sample_id < samples_per_seed
                for _ in 1:high_cfg.sample_stride
                    rk4_step!(u, high_cfg.dt, full_rhs!, params, work)
                    local_maxabs = max(local_maxabs, maximum(abs, u))
                    if !all(isfinite, u) || local_maxabs > 1.0e6
                        error("Observed stationary-reference simulation became unstable.")
                    end
                end
            end
        end
        done_count = Threads.atomic_add!(completed_seeds, 1) + 1
        if done_count == nseed || done_count % report_stride == 0
            log_message(
                "fit_score",
                @sprintf("Stationary reference progress: completed %d/%d seed trajectories", done_count, nseed),
            )
        end
    end

    log_message("fit_score", @sprintf("Stationary reference finished with %d samples", target_samples))
    return stationary[:, 1:target_samples]
end

function stationary_reference_cache_payload(rom_cfg::RomConfig, high_hash::String)
    return Dict(
        "high_resolution_dataset_hash" => high_hash,
        "rom" => Dict(
            "candidate_rollout_dt" => rom_cfg.candidate_rollout_dt,
            "candidate_rollout_steps" => rom_cfg.candidate_rollout_steps,
            "candidate_rollout_burnin" => rom_cfg.candidate_rollout_burnin,
            "candidate_rollout_ensembles" => rom_cfg.candidate_rollout_ensembles,
        ),
    )
end

function stationary_reference_cache_exists(dir::String)
    return isfile(stationary_reference_h5_path(dir)) &&
        isfile(cache_manifest_path(dir)) &&
        isfile(cache_parameters_path(dir))
end

function save_stationary_reference_cache(dir::String,
                                         stationary::AbstractMatrix,
                                         cache_params::AbstractDict,
                                         dependencies::AbstractDict)
    mkpath(dir)
    h5open(stationary_reference_h5_path(dir), "w") do file
        file["stationary_reference"] = Matrix{Float64}(stationary)
    end
    manifest = Dict(
        "artifact_type" => "stationary_reference",
        "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "artifact_hash" => basename(dir),
        "dependencies" => dependencies,
    )
    write_toml_file(cache_manifest_path(dir), manifest)
    write_toml_file(cache_parameters_path(dir), cache_params)
    return nothing
end

function load_stationary_reference_cache(dir::String)
    h5open(stationary_reference_h5_path(dir), "r") do file
        return read(file["stationary_reference"])
    end
end

function observed_stationary_reference(data::SimulationData,
                                      params::ModelParameters,
                                      high_cfg::HighResolutionConfig,
                                      rom_cfg::RomConfig,
                                      roots::Union{Nothing,RootPaths}=nothing,
                                      high_hash::Union{Nothing,String}=nothing)
    if roots === nothing || high_hash === nothing
        log_message("fit_score", "Building stationary reference without cache context")
        return build_observed_stationary_reference(data, params, high_cfg, rom_cfg)
    end

    payload = stationary_reference_cache_payload(rom_cfg, high_hash)
    cache_hash = artifact_hash(payload)
    cache_dir = stationary_reference_cache_dir(roots, cache_hash)
    if stationary_reference_cache_exists(cache_dir)
        log_cache_status("stationary reference", true, cache_hash, cache_dir)
        return load_stationary_reference_cache(cache_dir)
    end

    log_cache_status("stationary reference", false, cache_hash, cache_dir)
    stationary = build_observed_stationary_reference(data, params, high_cfg, rom_cfg)
    save_stationary_reference_cache(
        cache_dir,
        stationary,
        payload,
        Dict("high_resolution_dataset_hash" => high_hash),
    )
    return stationary
end

function merge_pdf_reference(observed_diag, pdf_reference_samples::AbstractMatrix, rom_cfg::RomConfig)
    pdf_x = ScoreEstimation.estimate_pdf_histogram(
        vec(pdf_reference_samples[1, :]);
        nbins=rom_cfg.pdf_bins,
        x_range=observed_diag.x_range,
    )
    pdf_y = ScoreEstimation.estimate_pdf_histogram(
        vec(pdf_reference_samples[2, :]);
        nbins=rom_cfg.pdf_bins,
        x_range=observed_diag.y_range,
    )
    pdf_xy = ScoreEstimation.estimate_bivariate_pdf_histogram(
        vec(pdf_reference_samples[1, :]),
        vec(pdf_reference_samples[2, :]);
        nbins=rom_cfg.joint_pdf_bins,
        x_range=observed_diag.x_range,
        y_range=observed_diag.y_range,
    )
    return merge(observed_diag, (pdf_x=pdf_x, pdf_y=pdf_y, pdf_xy=pdf_xy))
end

function build_observed_diagnostics(high_dataset,
                                    low_dataset,
                                    high_cfg::HighResolutionConfig,
                                    rom_cfg::RomConfig,
                                    roots::Union{Nothing,RootPaths}=nothing,
                                    high_hash::Union{Nothing,String}=nothing)
    log_message("fit_score", "Building observed diagnostics")
    observed = observed_matrix(high_dataset.data)
    low_obs = observed_matrix(low_dataset.data)
    stationary_reference = observed_stationary_reference(
        high_dataset.data,
        high_dataset.params,
        high_cfg,
        rom_cfg,
        roots,
        high_hash,
    )
    high_diag = ScoreEstimation.build_brusselator_diagnostics(
        build_diag_config(rom_cfg),
        observed,
        stationary_reference;
        sample_dt=high_dataset.data.sample_dt,
        pdf_bins=rom_cfg.pdf_bins,
        joint_pdf_bins=rom_cfg.joint_pdf_bins,
    )
    observed_diag = merge_pdf_reference(high_diag, low_obs, rom_cfg)
    log_message(
        "fit_score",
        @sprintf(
            "Observed diagnostics ready: observed=%d samples, low_res=%d samples, stationary_reference=%d samples",
            size(observed, 2),
            size(low_obs, 2),
            size(stationary_reference, 2),
        ),
    )
    return observed, low_obs, stationary_reference, observed_diag
end

function train_score_model(train_samples::AbstractMatrix, noise_scale::Float64, cfg::ScoreConfig)
    if cfg.use_gpu
        ScoreEstimation._cuda_module()
        ScoreEstimation._cuda_functional()
    end
    normalized, μ, σ = standardize_samples(train_samples)
    nn, losses, _, _, _, _, _ = ScoreEstimation.train(
        normalized;
        preprocessing=false,
        σ=noise_scale,
        neurons=cfg.neurons,
        n_epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.learning_rate,
        use_gpu=cfg.use_gpu,
        verbose=true,
        moment_weight_mean=cfg.mean_regularization,
        moment_weight_stein=cfg.stein_regularization,
        max_batches_per_epoch=cfg.max_batches_per_epoch,
    )
    log_message(
        "fit_score",
        @sprintf(
            "Score training finished for σ_noise=%.6f with %d epochs and final loss %.6e",
            noise_scale,
            length(losses),
            isempty(losses) ? NaN32 : losses[end],
        ),
    )
    return ScoreEstimation.ScoreModel(nn, noise_scale; mean=μ, scale=σ), losses
end

function rollout_reduced_model(score_model,
                               phi::AbstractMatrix,
                               sigma::AbstractMatrix,
                               stationary_samples::AbstractMatrix,
                               observed_dt::Real,
                               rollout_dt::Real,
                               n_steps::Int,
                               burn_in::Int,
                               n_ensembles::Int,
                               rng::AbstractRNG,
                               seed::Int;
                               device::Symbol=:cpu)
    resolution = snapshot_resolution(observed_dt, rollout_dt)
    x0 = sample_initial_conditions(stationary_samples, n_ensembles, rng)
    traj = ScoreEstimation.evolve_affine_score_langevin_snapshots(
        score_model,
        x0,
        phi,
        sigma;
        dt=rollout_dt,
        n_steps=n_steps,
        burn_in=burn_in,
        resolution=resolution,
        device=device,
        seed=seed,
        progress=true,
        progress_desc="Reduced Langevin rollout",
    )
    representative = select_representative_trajectory(traj)
    stationary = reshape(traj, size(traj, 1), :)
    return representative, stationary, Array(traj)
end

function select_representative_trajectory(snapshots::Array{Float32,3})
    best_idx = 1
    best_score = -Inf
    for ens in 1:size(snapshots, 3)
        traj = @view snapshots[:, :, ens]
        all(isfinite, traj) || continue
        variability = std(Float64.(traj[1, :])) + std(Float64.(traj[2, :]))
        if isfinite(variability) && variability > best_score
            best_score = variability
            best_idx = ens
        end
    end
    return copy(@view snapshots[:, :, best_idx])
end

function finite_stationary_samples(stationary::AbstractMatrix)
    mask = vec(all(isfinite, stationary; dims=1))
    any(mask) || error("No finite stationary ROM samples available.")
    return Matrix{Float64}(stationary[:, mask])
end

function valid_ensemble_diagnostic(diag)
    return !isempty(diag.acf_x) &&
        !isempty(diag.acf_y) &&
        !isempty(diag.ccf) &&
        all(isfinite, diag.acf_x) &&
        all(isfinite, diag.acf_y) &&
        all(isfinite, diag.ccf) &&
        diag.acf_x[1] > 0.0 &&
        diag.acf_y[1] > 0.0
end

function build_snapshot_diagnostics(diag_cfg,
                                    representative::AbstractMatrix,
                                    snapshots::Array{Float32,3},
                                    stationary::AbstractMatrix;
                                    sample_dt::Real,
                                    x_range,
                                    y_range,
                                    pdf_bins::Int,
                                    joint_pdf_bins::Int)
    stationary_finite = finite_stationary_samples(stationary)
    base = ScoreEstimation.build_brusselator_diagnostics(
        diag_cfg,
        representative,
        stationary_finite;
        sample_dt=sample_dt,
        x_range=x_range,
        y_range=y_range,
        pdf_bins=pdf_bins,
        joint_pdf_bins=joint_pdf_bins,
    )
    nens = size(snapshots, 3)
    acf_x_chunks = [zeros(Float64, length(base.acf_x)) for _ in 1:nthreads()]
    acf_y_chunks = [zeros(Float64, length(base.acf_y)) for _ in 1:nthreads()]
    ccf_chunks = [zeros(Float64, length(base.ccf)) for _ in 1:nthreads()]
    valid_counts = zeros(Int, nthreads())
    @threads for ens in 1:nens
        traj = @view snapshots[:, :, ens]
        all(isfinite, traj) || continue
        diag = ScoreEstimation.build_brusselator_diagnostics(
            diag_cfg,
            traj,
            stationary_finite;
            sample_dt=sample_dt,
            x_range=x_range,
            y_range=y_range,
            pdf_bins=pdf_bins,
            joint_pdf_bins=joint_pdf_bins,
        )
        valid_ensemble_diagnostic(diag) || continue
        tid = threadid()
        acf_x_chunks[tid] .+= diag.acf_x
        acf_y_chunks[tid] .+= diag.acf_y
        ccf_chunks[tid] .+= diag.ccf
        valid_counts[tid] += 1
    end
    valid_total = sum(valid_counts)
    if valid_total == 0
        base_acf_x = collect(Float64.(base.acf_x))
        base_acf_y = collect(Float64.(base.acf_y))
        base_ccf = collect(Float64.(base.ccf))
        !isempty(base_acf_x) && (base_acf_x[1] = 1.0)
        !isempty(base_acf_y) && (base_acf_y[1] = 1.0)
        return merge(base, (acf_x=base_acf_x, acf_y=base_acf_y, ccf=base_ccf, valid_ensemble_count=0))
    end
    acf_x = reduce(+, acf_x_chunks) ./ valid_total
    acf_y = reduce(+, acf_y_chunks) ./ valid_total
    ccf = reduce(+, ccf_chunks) ./ valid_total
    acf_x[1] = 1.0
    acf_y[1] = 1.0
    return merge(base, (acf_x=acf_x, acf_y=acf_y, ccf=ccf, valid_ensemble_count=valid_total))
end

function score_objective(metrics, stein_error::Real)
    return 3.0 * metrics.kl_xy +
        0.75 * (metrics.kl_x + metrics.kl_y) +
        0.5 * metrics.l1_xy +
        1.5 * Float64(stein_error)
end

function evaluate_score_model(noise_scale::Float64,
                              training_samples::AbstractMatrix,
                              score_model,
                              stationary_reference::AbstractMatrix,
                              observed_diag,
                              observed_dt::Float64,
                              score_cfg::ScoreConfig,
                              rom_cfg::RomConfig,
                              rng::AbstractRNG,
                              seed::Int)
    log_message(
        "fit_score",
        @sprintf(
            "Evaluating score model: σ_noise=%.6f, rollout_steps=%d, ensembles=%d",
            noise_scale,
            rom_cfg.candidate_rollout_steps,
            rom_cfg.candidate_rollout_ensembles,
        ),
    )
    identity = Matrix{Float64}(I, size(stationary_reference, 1), size(stationary_reference, 1))
    trajectory, stationary, snapshots = rollout_reduced_model(
        score_model,
        identity,
        identity,
        stationary_reference,
        observed_dt,
        rom_cfg.candidate_rollout_dt,
        rom_cfg.candidate_rollout_steps,
        rom_cfg.candidate_rollout_burnin,
        rom_cfg.candidate_rollout_ensembles,
        rng,
        seed,
        device=score_cfg.rollout_device,
    )
    log_message("fit_score", "Score rollout completed; building diagnostics")
    diagnostics = build_snapshot_diagnostics(
        build_diag_config(rom_cfg),
        trajectory,
        snapshots,
        stationary;
        sample_dt=observed_dt,
        x_range=observed_diag.x_range,
        y_range=observed_diag.y_range,
        pdf_bins=rom_cfg.pdf_bins,
        joint_pdf_bins=rom_cfg.joint_pdf_bins,
    )
    metrics = ScoreEstimation.compare_brusselator_diagnostics(observed_diag, diagnostics)
    log_message("fit_score", "Score diagnostics completed; estimating Stein matrix")
    stein_matrix = ScoreEstimation.compute_score_position_matrix(
        training_samples,
        score_model;
        batch_size=4 * score_cfg.batch_size,
        use_gpu=score_cfg.use_gpu,
    )
    stein_error = norm(stein_matrix + Matrix{Float64}(I, size(stein_matrix, 1), size(stein_matrix, 2)))
    log_message(
        "fit_score",
        @sprintf(
            "Score evaluation finished: objective=%.6f, KL_joint=%.6f, Stein=%.6f",
            score_objective(metrics, stein_error),
            metrics.kl_xy,
            stein_error,
        ),
    )
    return ScoreCandidate(
        noise_scale=noise_scale,
        score_model=score_model,
        losses=Float32[],
        diagnostics=diagnostics,
        metrics=metrics,
        stein_matrix=stein_matrix,
        stein_error=stein_error,
        objective=score_objective(metrics, stein_error),
        trajectory=trajectory,
        snapshots=snapshots,
        stationary=stationary,
    )
end

function evaluate_score_candidate(noise_scale::Float64,
                                  training_samples::AbstractMatrix,
                                  stationary_reference::AbstractMatrix,
                                  observed_diag,
                                  observed_dt::Float64,
                                  score_cfg::ScoreConfig,
                                  rom_cfg::RomConfig,
                                  rng::AbstractRNG,
                                  seed::Int)
    log_message("fit_score", @sprintf("Training score model for σ_noise = %.6f", noise_scale))
    score_model, losses = train_score_model(training_samples, noise_scale, score_cfg)
    candidate = evaluate_score_model(
        noise_scale,
        training_samples,
        score_model,
        stationary_reference,
        observed_diag,
        observed_dt,
        score_cfg,
        rom_cfg,
        rng,
        seed,
    )
    return ScoreCandidate(
        noise_scale=candidate.noise_scale,
        score_model=candidate.score_model,
        losses=losses,
        diagnostics=candidate.diagnostics,
        metrics=candidate.metrics,
        stein_matrix=candidate.stein_matrix,
        stein_error=candidate.stein_error,
        objective=candidate.objective,
        trajectory=candidate.trajectory,
        snapshots=candidate.snapshots,
        stationary=candidate.stationary,
    )
end

select_best_score_candidate(candidates::Vector{ScoreCandidate}) = candidates[argmin(getfield.(candidates, :objective))]

function write_vector_csv(path::String, columns::AbstractVector...)
    data = hcat(columns...)
    writedlm(path, data, ',')
    return nothing
end

function write_matrix_csv(path::String, matrix::AbstractMatrix)
    writedlm(path, matrix, ',')
    return nothing
end

function score_grid(score_model, x_range, y_range, n::Int)
    xs = collect(range(x_range[1], x_range[2], length=n))
    ys = collect(range(y_range[1], y_range[2], length=n))
    points = Matrix{Float32}(undef, 2, length(xs) * length(ys))
    idx = 1
    for y in ys, x in xs
        points[1, idx] = Float32(x)
        points[2, idx] = Float32(y)
        idx += 1
    end
    use_gpu = hasproperty(score_model, :model) &&
        hasproperty(score_model.model, :layers) &&
        !isempty(score_model.model.layers) &&
        hasproperty(first(score_model.model.layers), :weight) &&
        !(first(score_model.model.layers).weight isa Array)
    eval_points = use_gpu ? ScoreEstimation._to_device(points; use_gpu=true) : points
    score = Array(score_model(eval_points))
    rows = Matrix{Float64}(undef, length(xs) * length(ys), 5)
    idx = 1
    for y in ys, x in xs
        s1 = Float64(score[1, idx])
        s2 = Float64(score[2, idx])
        rows[idx, 1] = x
        rows[idx, 2] = y
        rows[idx, 3] = s1
        rows[idx, 4] = s2
        rows[idx, 5] = hypot(s1, s2)
        idx += 1
    end
    return xs, ys, rows
end

function score_smoothness_metrics(score_rows::AbstractMatrix, nx::Int, ny::Int)
    s1 = reshape(score_rows[:, 3], ny, nx)
    s2 = reshape(score_rows[:, 4], ny, nx)
    xs = sort(unique(vec(score_rows[:, 1])))
    ys = sort(unique(vec(score_rows[:, 2])))
    dx = nx > 1 ? xs[2] - xs[1] : 1.0
    dy = ny > 1 ? ys[2] - ys[1] : 1.0
    ds1_dx = nx > 1 ? diff(s1; dims=2) ./ dx : zeros(Float64, ny, 0)
    ds1_dy = ny > 1 ? diff(s1; dims=1) ./ dy : zeros(Float64, 0, nx)
    ds2_dx = nx > 1 ? diff(s2; dims=2) ./ dx : zeros(Float64, ny, 0)
    ds2_dy = ny > 1 ? diff(s2; dims=1) ./ dy : zeros(Float64, 0, nx)
    jump_x = nx > 1 ? sqrt.(diff(s1; dims=2).^2 .+ diff(s2; dims=2).^2) : zeros(Float64, ny, 0)
    jump_y = ny > 1 ? sqrt.(diff(s1; dims=1).^2 .+ diff(s2; dims=1).^2) : zeros(Float64, 0, nx)
    max_jump = max(isempty(jump_x) ? 0.0 : maximum(jump_x), isempty(jump_y) ? 0.0 : maximum(jump_y))
    jacobian_terms = Float64[]
    for block in (ds1_dx, ds1_dy, ds2_dx, ds2_dy)
        isempty(block) || append!(jacobian_terms, vec(abs.(block)))
    end
    curvature_terms = Float64[]
    nx > 2 && append!(curvature_terms, vec(abs.(diff(diff(s1; dims=2); dims=2)) ./ (dx^2)))
    nx > 2 && append!(curvature_terms, vec(abs.(diff(diff(s2; dims=2); dims=2)) ./ (dx^2)))
    ny > 2 && append!(curvature_terms, vec(abs.(diff(diff(s1; dims=1); dims=1)) ./ (dy^2)))
    ny > 2 && append!(curvature_terms, vec(abs.(diff(diff(s2; dims=1); dims=1)) ./ (dy^2)))
    return (
        max_neighbor_jump=max_jump,
        mean_jacobian_abs=isempty(jacobian_terms) ? 0.0 : mean(jacobian_terms),
        max_jacobian_abs=isempty(jacobian_terms) ? 0.0 : maximum(jacobian_terms),
        mean_curvature_abs=isempty(curvature_terms) ? 0.0 : mean(curvature_terms),
        max_curvature_abs=isempty(curvature_terms) ? 0.0 : maximum(curvature_terms),
    )
end

function write_metric_text(path::String, lines::Vector{String})
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return nothing
end

function moving_average(values::AbstractVector{<:Real}, radius::Int)
    n = length(values)
    data = Float64.(values)
    out = similar(data)
    for idx in 1:n
        lo = max(1, idx - radius)
        hi = min(n, idx + radius)
        out[idx] = mean(@view data[lo:hi])
    end
    return out
end

safe_relative_drop(old::Real, new::Real) = (Float64(old) - Float64(new)) / max(abs(Float64(old)), eps(Float64))

function training_diagnostics(losses::AbstractVector{<:Real})
    n = length(losses)
    if n == 0
        return (
            epochs=Float64[1.0],
            raw=Float64[NaN],
            smoothed=Float64[NaN],
            best=Float64[NaN],
            rel_change_pct=Float64[NaN],
            best_epoch=0,
            recommendation="Training loss history is unavailable for this artifact.",
            summary_lines=String[
                "Training loss history unavailable.",
                "If this run loaded an older cache, rerun fit_score with force=true to regenerate the loss history.",
                "This figure diagnoses optimization convergence only; it does not replace a validation set.",
            ],
        )
    end

    raw = Float64.(losses)
    epochs = Float64.(collect(1:n))
    smooth_radius = max(1, cld(n, 25))
    smoothed = moving_average(raw, smooth_radius)
    best = accumulate(min, raw)
    rel_change_pct = zeros(Float64, n)
    rel_change_pct[1] = NaN
    for idx in 2:n
        rel_change_pct[idx] = 100.0 * safe_relative_drop(raw[idx - 1], raw[idx])
    end

    best_epoch = argmin(raw)
    recent_window = clamp(max(5, cld(n, 10)), 1, n)
    recent_start = max(1, n - recent_window + 1)
    previous_start = max(1, recent_start - recent_window)
    previous_stop = max(recent_start - 1, previous_start)
    window_drop = previous_stop >= previous_start ?
        safe_relative_drop(mean(@view smoothed[previous_start:previous_stop]), mean(@view smoothed[recent_start:end])) :
        safe_relative_drop(smoothed[1], smoothed[end])
    final_gap = (raw[end] - raw[best_epoch]) / max(abs(raw[best_epoch]), eps(Float64))
    best_near_end = best_epoch >= recent_start

    recommendation = if n < 8
        "Too few epochs to judge convergence reliably."
    elseif final_gap > 0.02 && !best_near_end
        "Best training loss occurred well before the final epoch; fewer epochs or a smaller learning rate may be better."
    elseif window_drop > 0.05
        "Training loss is still decreasing materially near the end; more epochs may help."
    elseif window_drop < 0.005
        "Training loss has largely plateaued; more epochs are unlikely to change the model much."
    else
        "Training is still improving, but only slowly; the current epoch count is close to sufficient."
    end

    summary_lines = String[
        @sprintf("Epochs = %d", n),
        @sprintf("Initial loss = %.6e", raw[1]),
        @sprintf("Final loss = %.6e", raw[end]),
        @sprintf("Best loss = %.6e at epoch %d", raw[best_epoch], best_epoch),
        @sprintf("Final - best gap = %.2f%%", 100 * final_gap),
        @sprintf("Relative improvement over last %d epochs = %.2f%%", n - recent_start + 1, 100 * window_drop),
        "Assessment: " * recommendation,
        "This figure diagnoses optimization convergence only; it does not replace a validation set.",
    ]

    return (
        epochs=epochs,
        raw=raw,
        smoothed=smoothed,
        best=best,
        rel_change_pct=rel_change_pct,
        best_epoch=best_epoch,
        recommendation=recommendation,
        summary_lines=summary_lines,
    )
end

function write_training_plot_data(dir::String,
                                  candidate::ScoreCandidate,
                                  metrics_path::String)
    mkpath(dir)
    diagnostics = training_diagnostics(candidate.losses)
    write_vector_csv(
        joinpath(dir, "training_losses.csv"),
        diagnostics.epochs,
        diagnostics.raw,
        diagnostics.smoothed,
        diagnostics.best,
        diagnostics.rel_change_pct,
    )
    lines = vcat(
        diagnostics.summary_lines,
        String[
            @sprintf("Selected σ_noise = %.6f", candidate.noise_scale),
            @sprintf("Final score objective = %.6f", candidate.objective),
            @sprintf("KL(joint) = %.6f", candidate.metrics.kl_xy),
            @sprintf("Stein error = %.6f", candidate.stein_error),
        ],
    )
    write_metric_text(metrics_path, lines)
    return nothing
end

function _save_pdf_estimate(group, name::String, pdf)
    pdf_group = create_group(group, name)
    pdf_group["x"] = pdf.x
    pdf_group["density"] = pdf.density
    return nothing
end

function _save_bivariate_pdf_estimate(group, name::String, pdf)
    pdf_group = create_group(group, name)
    pdf_group["x"] = pdf.x
    pdf_group["y"] = pdf.y
    pdf_group["density"] = pdf.density
    return nothing
end

function _load_pdf_estimate(group, name::String)
    pdf_group = group[name]
    return ScoreEstimation.PDFEstimate(read(pdf_group["x"]), read(pdf_group["density"]))
end

function _load_bivariate_pdf_estimate(group, name::String)
    pdf_group = group[name]
    return ScoreEstimation.BivariatePDFEstimate(
        read(pdf_group["x"]),
        read(pdf_group["y"]),
        read(pdf_group["density"]),
    )
end

function _save_brusselator_diagnostics(group, diagnostics)
    _save_pdf_estimate(group, "pdf_x", diagnostics.pdf_x)
    _save_pdf_estimate(group, "pdf_y", diagnostics.pdf_y)
    _save_bivariate_pdf_estimate(group, "pdf_xy", diagnostics.pdf_xy)
    group["lag_axis"] = diagnostics.lag_axis
    group["acf_x"] = diagnostics.acf_x
    group["acf_y"] = diagnostics.acf_y
    group["ccf_axis"] = diagnostics.ccf_axis
    group["ccf"] = diagnostics.ccf
    group["freq_x"] = diagnostics.freq_x
    group["psd_x"] = diagnostics.psd_x
    group["freq_y"] = diagnostics.freq_y
    group["psd_y"] = diagnostics.psd_y
    group["x_mean"] = [Float64(diagnostics.x_mean)]
    group["y_mean"] = [Float64(diagnostics.y_mean)]
    group["x_std"] = [Float64(diagnostics.x_std)]
    group["y_std"] = [Float64(diagnostics.y_std)]
    group["corr_xy"] = [Float64(diagnostics.corr_xy)]
    group["tau_x"] = [Float64(diagnostics.tau_x)]
    group["tau_y"] = [Float64(diagnostics.tau_y)]
    group["neff_x"] = [Float64(diagnostics.neff_x)]
    group["neff_y"] = [Float64(diagnostics.neff_y)]
    group["phase_count"] = [Int(diagnostics.phase_count)]
    group["x_range"] = collect(Float64.(diagnostics.x_range))
    group["y_range"] = collect(Float64.(diagnostics.y_range))
    if hasproperty(diagnostics, :valid_ensemble_count)
        group["valid_ensemble_count"] = [Int(diagnostics.valid_ensemble_count)]
    end
    return nothing
end

function _load_brusselator_diagnostics(group)
    diagnostics = (
        pdf_x=_load_pdf_estimate(group, "pdf_x"),
        pdf_y=_load_pdf_estimate(group, "pdf_y"),
        pdf_xy=_load_bivariate_pdf_estimate(group, "pdf_xy"),
        lag_axis=read(group["lag_axis"]),
        acf_x=read(group["acf_x"]),
        acf_y=read(group["acf_y"]),
        ccf_axis=read(group["ccf_axis"]),
        ccf=read(group["ccf"]),
        freq_x=read(group["freq_x"]),
        psd_x=read(group["psd_x"]),
        freq_y=read(group["freq_y"]),
        psd_y=read(group["psd_y"]),
        x_mean=read(group["x_mean"])[1],
        y_mean=read(group["y_mean"])[1],
        x_std=read(group["x_std"])[1],
        y_std=read(group["y_std"])[1],
        corr_xy=read(group["corr_xy"])[1],
        tau_x=read(group["tau_x"])[1],
        tau_y=read(group["tau_y"])[1],
        neff_x=read(group["neff_x"])[1],
        neff_y=read(group["neff_y"])[1],
        phase_count=Int(read(group["phase_count"])[1]),
        x_range=Tuple(Float64.(read(group["x_range"]))),
        y_range=Tuple(Float64.(read(group["y_range"]))),
    )
    if haskey(group, "valid_ensemble_count")
        return merge(diagnostics, (valid_ensemble_count=Int(read(group["valid_ensemble_count"])[1]),))
    end
    return diagnostics
end

function _save_brusselator_metrics(group, metrics)
    group["kl_x"] = [Float64(metrics.kl_x)]
    group["kl_y"] = [Float64(metrics.kl_y)]
    group["kl_xy"] = [Float64(metrics.kl_xy)]
    group["l1_x"] = [Float64(metrics.l1_x)]
    group["l1_y"] = [Float64(metrics.l1_y)]
    group["l1_xy"] = [Float64(metrics.l1_xy)]
    group["rmse_acf_x"] = [Float64(metrics.rmse_acf_x)]
    group["rmse_acf_y"] = [Float64(metrics.rmse_acf_y)]
    group["rmse_ccf"] = [Float64(metrics.rmse_ccf)]
    group["short_lag_points"] = [Int(metrics.short_lag_points)]
    group["short_ccf_points"] = [Int(metrics.short_ccf_points)]
    return nothing
end

function _load_brusselator_metrics(group)
    return (
        kl_x=read(group["kl_x"])[1],
        kl_y=read(group["kl_y"])[1],
        kl_xy=read(group["kl_xy"])[1],
        l1_x=read(group["l1_x"])[1],
        l1_y=read(group["l1_y"])[1],
        l1_xy=read(group["l1_xy"])[1],
        rmse_acf_x=read(group["rmse_acf_x"])[1],
        rmse_acf_y=read(group["rmse_acf_y"])[1],
        rmse_ccf=read(group["rmse_ccf"])[1],
        short_lag_points=Int(read(group["short_lag_points"])[1]),
        short_ccf_points=Int(read(group["short_ccf_points"])[1]),
    )
end

function write_score_plot_data(dir::String,
                               observed_diag,
                               observed_traj::AbstractMatrix,
                               observed_time::AbstractVector,
                               candidate::ScoreCandidate,
                               rom_cfg::RomConfig,
                               metrics_path::String;
                               pdf_reference_samples::AbstractMatrix)
    mkpath(dir)
    obs_pdf_x = ScoreEstimation.estimate_pdf_histogram(
        vec(pdf_reference_samples[1, :]);
        nbins=rom_cfg.pdf_bins,
        x_range=observed_diag.x_range,
    )
    obs_pdf_y = ScoreEstimation.estimate_pdf_histogram(
        vec(pdf_reference_samples[2, :]);
        nbins=rom_cfg.pdf_bins,
        x_range=observed_diag.y_range,
    )
    obs_pdf_xy = ScoreEstimation.estimate_bivariate_pdf_histogram(
        vec(pdf_reference_samples[1, :]),
        vec(pdf_reference_samples[2, :]);
        nbins=rom_cfg.joint_pdf_bins,
        x_range=observed_diag.x_range,
        y_range=observed_diag.y_range,
    )
    write_vector_csv(joinpath(dir, "pdf_x.csv"), obs_pdf_x.x, obs_pdf_x.density, candidate.diagnostics.pdf_x.density)
    write_vector_csv(joinpath(dir, "pdf_y.csv"), obs_pdf_y.x, obs_pdf_y.density, candidate.diagnostics.pdf_y.density)
    write_vector_csv(joinpath(dir, "acf_x.csv"), observed_diag.lag_axis, observed_diag.acf_x, candidate.diagnostics.acf_x)
    write_vector_csv(joinpath(dir, "acf_y.csv"), observed_diag.lag_axis, observed_diag.acf_y, candidate.diagnostics.acf_y)
    write_vector_csv(joinpath(dir, "ccf.csv"), observed_diag.ccf_axis, observed_diag.ccf, candidate.diagnostics.ccf)
    write_matrix_csv(joinpath(dir, "joint_obs_density.csv"), obs_pdf_xy.density)
    write_matrix_csv(joinpath(dir, "joint_rom_density.csv"), candidate.diagnostics.pdf_xy.density)
    write_vector_csv(joinpath(dir, "joint_x.csv"), obs_pdf_xy.x)
    write_vector_csv(joinpath(dir, "joint_y.csv"), obs_pdf_xy.y)
    n_traj = min(rom_cfg.x_excerpt_points, size(observed_traj, 2), size(candidate.trajectory, 2))
    obs_idx = (size(observed_traj, 2) - n_traj + 1):size(observed_traj, 2)
    rom_idx = (size(candidate.trajectory, 2) - n_traj + 1):size(candidate.trajectory, 2)
    traj_time = observed_time[obs_idx] .- observed_time[obs_idx[1]]
    write_vector_csv(joinpath(dir, "trajectory_obs.csv"), traj_time, observed_traj[1, obs_idx], observed_traj[2, obs_idx])
    write_vector_csv(joinpath(dir, "trajectory_rom.csv"), traj_time, candidate.trajectory[1, rom_idx], candidate.trajectory[2, rom_idx])
    write_matrix_csv(joinpath(dir, "stein_matrix.csv"), candidate.stein_matrix)
    write_matrix_csv(joinpath(dir, "phi_symmetric.csv"), Matrix{Float64}(I, 2, 2))
    write_matrix_csv(joinpath(dir, "phi_antisymmetric.csv"), zeros(Float64, 2, 2))
    write_matrix_csv(joinpath(dir, "phi.csv"), Matrix{Float64}(I, 2, 2))
    write_matrix_csv(joinpath(dir, "sigma.csv"), Matrix{Float64}(I, 2, 2))
    xs, ys, score_rows = score_grid(candidate.score_model, observed_diag.x_range, observed_diag.y_range, rom_cfg.score_grid_n)
    write_vector_csv(joinpath(dir, "score_x.csv"), xs)
    write_vector_csv(joinpath(dir, "score_y.csv"), ys)
    write_matrix_csv(joinpath(dir, "score_grid.csv"), score_rows)
    smoothness = score_smoothness_metrics(score_rows, length(xs), length(ys))
    write_metric_text(metrics_path, [
        @sprintf("KL(x1) = %.5f", candidate.metrics.kl_x),
        @sprintf("KL(x2) = %.5f", candidate.metrics.kl_y),
        @sprintf("KL(joint) = %.5f", candidate.metrics.kl_xy),
        @sprintf("L1(joint) = %.5f", candidate.metrics.l1_xy),
        @sprintf("RMSE ACF(x1) = %.5f", candidate.metrics.rmse_acf_x),
        @sprintf("RMSE ACF(x2) = %.5f", candidate.metrics.rmse_acf_y),
        @sprintf("RMSE CCF = %.5f", candidate.metrics.rmse_ccf),
        @sprintf("||V + I||_F = %.5f", candidate.stein_error),
        @sprintf("max |∇s| proxy = %.5f", smoothness.max_jacobian_abs),
        @sprintf("mean |∇s| proxy = %.5f", smoothness.mean_jacobian_abs),
        @sprintf("max neighbor jump = %.5f", smoothness.max_neighbor_jump),
        @sprintf("max curvature proxy = %.5f", smoothness.max_curvature_abs),
    ])
    return nothing
end

function score_cache_payload(score_cfg::ScoreConfig, low_hash::String)
    return Dict(
        "score" => Dict(
            "rng_seed" => score_cfg.rng_seed,
            "noise_scales" => score_cfg.noise_scales,
            "epochs" => score_cfg.epochs,
            "batch_size" => score_cfg.batch_size,
            "neurons" => score_cfg.neurons,
            "learning_rate" => score_cfg.learning_rate,
            "use_gpu" => score_cfg.use_gpu,
            "rollout_device" => String(score_cfg.rollout_device),
            "mean_regularization" => score_cfg.mean_regularization,
            "stein_regularization" => score_cfg.stein_regularization,
            "max_training_samples" => score_cfg.max_training_samples,
            "max_stationary_samples" => score_cfg.max_stationary_samples,
            "max_batches_per_epoch" => score_cfg.max_batches_per_epoch === nothing ? "nothing" : score_cfg.max_batches_per_epoch,
        ),
        "low_resolution_dataset_hash" => low_hash,
    )
end

function score_cache_exists(dir::String)
    return isfile(score_model_bson_path(dir)) &&
        isfile(score_summary_path(dir)) &&
        isfile(cache_manifest_path(dir)) &&
        isfile(cache_parameters_path(dir))
end

function score_noise_requested(score_cfg::ScoreConfig, noise_scale::Real)
    return any(isapprox(Float64(noise_scale), σ; atol=1.0e-12, rtol=1.0e-12) for σ in score_cfg.noise_scales)
end

function find_compatible_score_cache(roots::RootPaths, score_cfg::ScoreConfig, low_hash::String)
    scores_root = joinpath(roots.data_dir, "scores")
    isdir(scores_root) || return nothing

    best_match = nothing
    best_created_at = ""
    for entry in readdir(scores_root)
        dir = joinpath(scores_root, entry)
        isdir(dir) || continue
        score_cache_exists(dir) || continue

        manifest = TOML.parsefile(cache_manifest_path(dir))
        dependencies = get(manifest, "dependencies", Dict{String,Any}())
        get(dependencies, "low_resolution_dataset_hash", nothing) == low_hash || continue

        summary = TOML.parsefile(score_summary_path(dir))
        noise_scale = Float64(get(summary, "noise_scale", NaN))
        score_noise_requested(score_cfg, noise_scale) || continue

        created_at = String(get(manifest, "created_at", ""))
        if best_match === nothing || created_at > best_created_at
            best_match = (dir=dir, hash=basename(dir), summary=summary)
            best_created_at = created_at
        end
    end

    return best_match
end

score_evaluation_cache_exists(dir::String) = isfile(score_evaluation_h5_path(dir))

function placeholder_score_candidate(score_model, summary::AbstractDict)
    return ScoreCandidate(
        noise_scale=Float64(get(summary, "noise_scale", NaN)),
        score_model=score_model,
        losses=Float32[],
        diagnostics=nothing,
        metrics=nothing,
        stein_matrix=zeros(Float64, 2, 2),
        stein_error=Float64(get(summary, "score_stein_error", NaN)),
        objective=Float64(get(summary, "score_objective", NaN)),
        trajectory=Matrix{Float32}(undef, 2, 0),
        snapshots=Array{Float32,3}(undef, 2, 0, 0),
        stationary=Matrix{Float32}(undef, 2, 0),
    )
end

function save_score_evaluation_cache(dir::String, candidate::ScoreCandidate)
    mkpath(dir)
    h5open(score_evaluation_h5_path(dir), "w") do file
        file["noise_scale"] = [candidate.noise_scale]
        file["losses"] = candidate.losses
        file["trajectory"] = candidate.trajectory
        file["snapshots"] = candidate.snapshots
        file["stationary"] = candidate.stationary
        file["stein_matrix"] = candidate.stein_matrix
        file["stein_error"] = [candidate.stein_error]
        file["objective"] = [candidate.objective]
        diagnostics_group = create_group(file, "diagnostics")
        _save_brusselator_diagnostics(diagnostics_group, candidate.diagnostics)
        metrics_group = create_group(file, "metrics")
        _save_brusselator_metrics(metrics_group, candidate.metrics)
    end
    return nothing
end

function load_cached_score_candidate(dir::String, score_model)
    score_evaluation_cache_exists(dir) || return nothing
    h5open(score_evaluation_h5_path(dir), "r") do file
        return ScoreCandidate(
            noise_scale=read(file["noise_scale"])[1],
            score_model=score_model,
            losses=read(file["losses"]),
            diagnostics=_load_brusselator_diagnostics(file["diagnostics"]),
            metrics=_load_brusselator_metrics(file["metrics"]),
            stein_matrix=read(file["stein_matrix"]),
            stein_error=read(file["stein_error"])[1],
            objective=read(file["objective"])[1],
            trajectory=read(file["trajectory"]),
            snapshots=read(file["snapshots"]),
            stationary=read(file["stationary"]),
        )
    end
end

function save_score_cache(dir::String,
                          candidate::ScoreCandidate,
                          candidates::Vector{ScoreCandidate},
                          cache_params::Dict{String,Any},
                          dependencies::AbstractDict,
                          system_label::String)
    mkpath(dir)
    ScoreEstimation.save_model(
        candidate.score_model,
        score_model_bson_path(dir);
        metadata=Dict(
            "noise_scale" => candidate.noise_scale,
            "score_objective" => candidate.objective,
        ),
    )
    save_score_evaluation_cache(dir, candidate)
    score_summary = Dict(
        "system_label" => system_label,
        "noise_scale" => candidate.noise_scale,
        "score_objective" => candidate.objective,
        "score_kl_x1" => candidate.metrics.kl_x,
        "score_kl_x2" => candidate.metrics.kl_y,
        "score_kl_joint" => candidate.metrics.kl_xy,
        "score_l1_joint" => candidate.metrics.l1_xy,
        "score_acf_rmse_x1" => candidate.metrics.rmse_acf_x,
        "score_acf_rmse_x2" => candidate.metrics.rmse_acf_y,
        "score_ccf_rmse" => candidate.metrics.rmse_ccf,
        "score_stein_error" => candidate.stein_error,
        "candidate_noise_scales" => [cand.noise_scale for cand in candidates],
    )
    write_toml_file(score_summary_path(dir), score_summary)
    manifest = Dict(
        "artifact_type" => "score",
        "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "artifact_hash" => basename(dir),
        "dependencies" => dependencies,
    )
    write_toml_file(cache_manifest_path(dir), manifest)
    write_toml_file(cache_parameters_path(dir), cache_params)
    return nothing
end

function load_cached_score(dir::String)
    score_model, _ = ScoreEstimation.load_model(score_model_bson_path(dir))
    summary = TOML.parsefile(score_summary_path(dir))
    return score_model, summary
end
