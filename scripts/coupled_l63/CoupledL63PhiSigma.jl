Base.@kwdef struct ROMCandidate
    noise_scale::Float64
    score_model
    phi_raw::Matrix{Float64}
    phi_rom::Matrix{Float64}
    sigma_derived::Matrix{Float64}
    sigma_rom::Matrix{Float64}
    info::Dict{Symbol,Any}
    diagnostics
    metrics
    objective::Float64
    trajectory::Matrix{Float32}
    snapshots::Array{Float32,3}
    stationary::Matrix{Float32}
end

function candidate_objective(metrics)
    ccf_term = isfinite(metrics.rmse_ccf) ? metrics.rmse_ccf : 10.0
    return 2.0 * metrics.kl_xy + 0.5 * (metrics.kl_x + metrics.kl_y) +
        8.0 * (metrics.rmse_acf_x + metrics.rmse_acf_y) + 4.0 * ccf_term
end

function perturb_samples(samples::AbstractMatrix, score_model, σ::Real, rng::AbstractRNG)
    dim = size(samples, 1)
    scales = if hasproperty(score_model, :scale)
        Float32.(Float64(σ) .* Float64.(collect(getproperty(score_model, :scale))))
    else
        fill(Float32(σ), dim)
    end
    return Float32.(samples) .+ reshape(scales, :, 1) .* randn(rng, Float32, size(samples))
end

function sparse_triplets(matrix::SparseArrays.SparseMatrixCSC)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for col in 1:size(matrix, 2)
        for ptr in nzrange(matrix, col)
            push!(rows, rowvals(matrix)[ptr])
            push!(cols, col)
            push!(vals, nonzeros(matrix)[ptr])
        end
    end
    return rows, cols, vals
end

function raw_phi_and_sigma(correlated_samples::AbstractMatrix,
                           stationary_samples::AbstractMatrix,
                           score_model,
                           dt::Real,
                           cfg::PhiSigmaConfig,
                           batch_size::Int,
                           use_gpu::Bool,
                           rng::AbstractRNG)
    generator_σ = cfg.generator_perturb_scale > 0 ? cfg.generator_perturb_scale : Float64(score_model.noise_scale)
    stationary_σ = cfg.stationary_perturb_scale > 0 ? cfg.stationary_perturb_scale : Float64(score_model.noise_scale)
    log_message(
        "fit_phi_sigma",
        @sprintf(
            "Phi/Sigma estimation started: correlated_samples=%d, stationary_samples=%d, generator_σ=%.6f, stationary_σ=%.6f",
            size(correlated_samples, 2),
            size(stationary_samples, 2),
            generator_σ,
            stationary_σ,
        ),
    )
    correlated_noisy = perturb_samples(correlated_samples, score_model, generator_σ, rng)
    log_message("fit_phi_sigma", "Constructing state-space partition")
    ssp = StateSpacePartition(correlated_noisy; method=Tree(false, cfg.minimum_probability), override=cfg.partition_override)
    labels = ssp.partitions
    n_states = maximum(labels)
    log_message("fit_phi_sigma", @sprintf("Partition constructed with %d states", n_states))
    log_message("fit_phi_sigma", "Estimating sparse generator Q")
    Q = ScoreEstimation.sparse_generator(labels, n_states; dt=dt)
    log_message("fit_phi_sigma", "Computing cluster centers, invariant weights, and Cdot")
    centers, pi_vec, counts = ScoreEstimation.compute_cluster_centers_and_pi(correlated_noisy, labels, n_states)
    cdot = ScoreEstimation.compute_generator_cdot(centers, Q, pi_vec)
    stationary_noisy = perturb_samples(stationary_samples, score_model, stationary_σ, rng)
    log_message("fit_phi_sigma", "Estimating Stein matrix V")
    V = ScoreEstimation.compute_score_position_matrix(
        stationary_noisy,
        score_model;
        batch_size=4 * batch_size,
        use_gpu=use_gpu,
    )
    # Phi is defined as the raw coarse-grained estimator Cdot / V with no
    # post-processing, corrections, or rescaling applied to Phi itself.
    phi_raw = Float64.(cdot / V)
    phi_symmetric_raw = 0.5 .* (phi_raw + transpose(phi_raw))
    eig = eigen(Symmetric(phi_symmetric_raw))
    clipped = max.(eig.values, cfg.sigma_regularization)
    phi_symmetric = eig.vectors * Diagonal(clipped) * transpose(eig.vectors)
    sigma_derived = Matrix(cholesky(Symmetric(phi_symmetric)).L)
    log_message(
        "fit_phi_sigma",
        @sprintf("Phi/Sigma estimation finished: n_states=%d, min_eig_after_clip=%.6e", n_states, minimum(clipped)),
    )
    q_rows, q_cols, q_vals = sparse_triplets(Q)
    info = Dict{Symbol,Any}(
        :V => Float64.(V),
        :cdot => Float64.(cdot),
        :centers => Float64.(centers),
        :pi_vec => Float64.(pi_vec),
        :counts => Int.(counts),
        :q_rows => q_rows,
        :q_cols => q_cols,
        :q_vals => q_vals,
        :q_size => [size(Q, 1), size(Q, 2)],
        :minimum_probability => Float64(cfg.minimum_probability),
        :generator_perturb_scale => generator_σ,
        :stationary_perturb_scale => stationary_σ,
        :phi_symmetric_raw => phi_symmetric_raw,
        :phi_symmetric => phi_symmetric,
        :phi_antisymmetric => 0.5 .* (phi_raw - transpose(phi_raw)),
        :n_states => n_states,
    )
    return phi_raw, sigma_derived, info
end

function rom_matrices(phi_raw::AbstractMatrix, sigma_derived::AbstractMatrix, cfg::PhiSigmaConfig)
    phi_rom = cfg.use_identity_phi ? Matrix{Float64}(I, size(phi_raw, 1), size(phi_raw, 2)) : Matrix{Float64}(phi_raw)
    sigma_rom = cfg.use_identity_sigma ? Matrix{Float64}(I, size(sigma_derived, 1), size(sigma_derived, 2)) : Matrix{Float64}(sigma_derived)
    return phi_rom, sigma_rom
end

function build_rom_candidate(score_model,
                             noise_scale::Float64,
                             phi_raw::AbstractMatrix,
                             sigma_derived::AbstractMatrix,
                             info::Dict{Symbol,Any},
                             stationary_reference::AbstractMatrix,
                             observed_diag,
                             observed_dt::Float64,
                             score_cfg::ScoreConfig,
                             rom_cfg::RomConfig,
                             phi_cfg::PhiSigmaConfig,
                             rng::AbstractRNG,
                             seed::Int)
    log_message(
        "fit_phi_sigma",
        @sprintf(
            "Building ROM candidate: rollout_steps=%d, ensembles=%d, use_identity_phi=%s, use_identity_sigma=%s",
            rom_cfg.final_rollout_steps,
            rom_cfg.final_rollout_ensembles,
            string(phi_cfg.use_identity_phi),
            string(phi_cfg.use_identity_sigma),
        ),
    )
    phi_rom, sigma_rom = rom_matrices(phi_raw, sigma_derived, phi_cfg)
    trajectory, stationary, snapshots = rollout_reduced_model(
        score_model,
        phi_rom,
        sigma_rom,
        stationary_reference,
        observed_dt,
        rom_cfg.final_rollout_dt,
        rom_cfg.final_rollout_steps,
        rom_cfg.final_rollout_burnin,
        rom_cfg.final_rollout_ensembles,
        rng,
        seed,
        device=score_cfg.rollout_device,
    )
    log_message("fit_phi_sigma", "ROM rollout completed; building diagnostics")
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
    log_message(
        "fit_phi_sigma",
        @sprintf("ROM diagnostics finished: KL_joint=%.6f, objective=%.6f", metrics.kl_xy, candidate_objective(metrics)),
    )
    return ROMCandidate(
        noise_scale=noise_scale,
        score_model=score_model,
        phi_raw=Matrix{Float64}(phi_raw),
        phi_rom=phi_rom,
        sigma_derived=Matrix{Float64}(sigma_derived),
        sigma_rom=sigma_rom,
        info=info,
        diagnostics=diagnostics,
        metrics=metrics,
        objective=candidate_objective(metrics),
        trajectory=trajectory,
        snapshots=snapshots,
        stationary=stationary,
    )
end

function write_rom_plot_data(dir::String,
                             observed_diag,
                             observed_traj::AbstractMatrix,
                             observed_time::AbstractVector,
                             candidate::ROMCandidate,
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
    write_matrix_csv(joinpath(dir, "stein_matrix.csv"), Float64.(candidate.info[:V]))
    write_matrix_csv(joinpath(dir, "phi_symmetric.csv"), Float64.(candidate.info[:phi_symmetric]))
    write_matrix_csv(joinpath(dir, "phi_antisymmetric.csv"), Float64.(candidate.info[:phi_antisymmetric]))
    write_matrix_csv(joinpath(dir, "phi.csv"), candidate.phi_rom)
    write_matrix_csv(joinpath(dir, "sigma.csv"), candidate.sigma_rom)
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
        @sprintf("clusters = %d", Int(candidate.info[:n_states])),
        @sprintf("max |∇s| proxy = %.5f", smoothness.max_jacobian_abs),
        @sprintf("mean |∇s| proxy = %.5f", smoothness.mean_jacobian_abs),
        @sprintf("max neighbor jump = %.5f", smoothness.max_neighbor_jump),
        @sprintf("max curvature proxy = %.5f", smoothness.max_curvature_abs),
    ])
    return nothing
end

function phi_sigma_cache_payload(phi_cfg::PhiSigmaConfig, score_hash::String, high_hash::String)
    return Dict(
        "phi_sigma" => Dict(
            "minimum_probability" => phi_cfg.minimum_probability,
            "partition_override" => phi_cfg.partition_override,
            "generator_perturb_scale" => phi_cfg.generator_perturb_scale,
            "stationary_perturb_scale" => phi_cfg.stationary_perturb_scale,
            "sigma_regularization" => phi_cfg.sigma_regularization,
            "use_identity_phi" => phi_cfg.use_identity_phi,
            "use_identity_sigma" => phi_cfg.use_identity_sigma,
            "max_correlated_samples" => phi_cfg.max_correlated_samples,
        ),
        "score_hash" => score_hash,
        "high_resolution_dataset_hash" => high_hash,
    )
end

function phi_sigma_cache_exists(dir::String)
    return isfile(phi_sigma_h5_path(dir)) &&
        isfile(phi_sigma_summary_path(dir)) &&
        isfile(cache_manifest_path(dir)) &&
        isfile(cache_parameters_path(dir))
end

function placeholder_rom_candidate(score_model, cached)
    summary = cached.summary
    return ROMCandidate(
        noise_scale=Float64(get(summary, "noise_scale", NaN)),
        score_model=score_model,
        phi_raw=cached.phi_raw,
        phi_rom=cached.phi_rom,
        sigma_derived=cached.sigma_derived,
        sigma_rom=cached.sigma_rom,
        info=cached.info,
        diagnostics=nothing,
        metrics=nothing,
        objective=Float64(get(summary, "objective", NaN)),
        trajectory=Matrix{Float32}(undef, 2, 0),
        snapshots=Array{Float32,3}(undef, 2, 0, 0),
        stationary=Matrix{Float32}(undef, 2, 0),
    )
end

function save_phi_sigma_cache(dir::String,
                              candidate::ROMCandidate,
                              cache_params::Dict{String,Any},
                              dependencies::AbstractDict,
                              system_label::String)
    mkpath(dir)
    h5open(phi_sigma_h5_path(dir), "w") do file
        file["phi_raw"] = candidate.phi_raw
        file["phi_rom"] = candidate.phi_rom
        file["sigma_derived"] = candidate.sigma_derived
        file["sigma_rom"] = candidate.sigma_rom
        file["phi_symmetric_raw"] = Float64.(candidate.info[:phi_symmetric_raw])
        file["phi_symmetric"] = Float64.(candidate.info[:phi_symmetric])
        file["phi_antisymmetric"] = Float64.(candidate.info[:phi_antisymmetric])
        file["stein_matrix"] = Float64.(candidate.info[:V])
        file["cdot"] = Float64.(candidate.info[:cdot])
        file["centers"] = Float64.(candidate.info[:centers])
        file["pi_vec"] = Float64.(candidate.info[:pi_vec])
        file["counts"] = Int.(candidate.info[:counts])
        file["q_rows"] = Int.(candidate.info[:q_rows])
        file["q_cols"] = Int.(candidate.info[:q_cols])
        file["q_vals"] = Float64.(candidate.info[:q_vals])
        file["q_size"] = Int.(candidate.info[:q_size])
        file["n_states"] = [Int(candidate.info[:n_states])]
        file["minimum_probability"] = [Float64(candidate.info[:minimum_probability])]
        file["generator_perturb_scale"] = [Float64(candidate.info[:generator_perturb_scale])]
        file["stationary_perturb_scale"] = [Float64(candidate.info[:stationary_perturb_scale])]
        rom_group = create_group(file, "rom_evaluation")
        rom_group["noise_scale"] = [candidate.noise_scale]
        rom_group["objective"] = [candidate.objective]
        rom_group["trajectory"] = candidate.trajectory
        rom_group["snapshots"] = candidate.snapshots
        rom_group["stationary"] = candidate.stationary
        diagnostics_group = create_group(rom_group, "diagnostics")
        _save_brusselator_diagnostics(diagnostics_group, candidate.diagnostics)
        metrics_group = create_group(rom_group, "metrics")
        _save_brusselator_metrics(metrics_group, candidate.metrics)
    end
    summary = Dict(
        "system_label" => system_label,
        "objective" => candidate.objective,
        "noise_scale" => candidate.noise_scale,
        "kl_x1" => candidate.metrics.kl_x,
        "kl_x2" => candidate.metrics.kl_y,
        "kl_joint" => candidate.metrics.kl_xy,
        "acf_rmse_x1" => candidate.metrics.rmse_acf_x,
        "acf_rmse_x2" => candidate.metrics.rmse_acf_y,
        "ccf_rmse" => candidate.metrics.rmse_ccf,
        "n_states" => Int(candidate.info[:n_states]),
    )
    write_toml_file(phi_sigma_summary_path(dir), summary)
    manifest = Dict(
        "artifact_type" => "phi_sigma",
        "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "artifact_hash" => basename(dir),
        "dependencies" => dependencies,
    )
    write_toml_file(cache_manifest_path(dir), manifest)
    write_toml_file(cache_parameters_path(dir), cache_params)
    return nothing
end

function load_cached_phi_sigma(dir::String)
    h5open(phi_sigma_h5_path(dir), "r") do file
        info = Dict{Symbol,Any}(
            :V => read(file["stein_matrix"]),
            :cdot => read(file["cdot"]),
            :centers => read(file["centers"]),
            :pi_vec => read(file["pi_vec"]),
            :counts => read(file["counts"]),
            :q_rows => read(file["q_rows"]),
            :q_cols => read(file["q_cols"]),
            :q_vals => read(file["q_vals"]),
            :q_size => read(file["q_size"]),
            :n_states => read(file["n_states"])[1],
            :minimum_probability => read(file["minimum_probability"])[1],
            :generator_perturb_scale => read(file["generator_perturb_scale"])[1],
            :stationary_perturb_scale => read(file["stationary_perturb_scale"])[1],
            :phi_symmetric_raw => read(file["phi_symmetric_raw"]),
            :phi_symmetric => read(file["phi_symmetric"]),
            :phi_antisymmetric => read(file["phi_antisymmetric"]),
        )
        evaluation = if haskey(file, "rom_evaluation")
            rom_group = file["rom_evaluation"]
            (
                noise_scale=read(rom_group["noise_scale"])[1],
                objective=read(rom_group["objective"])[1],
                diagnostics=_load_brusselator_diagnostics(rom_group["diagnostics"]),
                metrics=_load_brusselator_metrics(rom_group["metrics"]),
                trajectory=read(rom_group["trajectory"]),
                snapshots=read(rom_group["snapshots"]),
                stationary=read(rom_group["stationary"]),
            )
        else
            nothing
        end
        return (
            phi_raw=read(file["phi_raw"]),
            phi_rom=read(file["phi_rom"]),
            sigma_derived=read(file["sigma_derived"]),
            sigma_rom=read(file["sigma_rom"]),
            info=info,
            evaluation=evaluation,
            summary=TOML.parsefile(phi_sigma_summary_path(dir)),
        )
    end
end
