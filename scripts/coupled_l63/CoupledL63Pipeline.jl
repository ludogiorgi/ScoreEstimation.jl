module CoupledL63Pipeline

using Dates
using DelimitedFiles
using Flux
using HDF5
using LinearAlgebra
using Printf
using Random
using ScoreEstimation
using SHA
using SparseArrays
using StateSpacePartitions
using Statistics
using TOML
using Base.Threads

const SCRIPT_DIR = @__DIR__
const DEFAULT_CONFIG = joinpath(SCRIPT_DIR, "parameters.toml")
const STAGE_SEQUENCE = (:generate_data, :fit_score, :fit_phi_sigma, :estimate_residuals)
const LOG_TIMESTAMP_FORMAT = dateformat"yyyy-mm-dd HH:MM:SS"

timestamp_now() = Dates.format(now(), LOG_TIMESTAMP_FORMAT)

function log_message(scope::AbstractString, message::AbstractString)
    println("[" * timestamp_now() * "] [" * scope * "] " * message)
    flush(stdout)
    return nothing
end

function format_elapsed(seconds::Real)
    if seconds < 60
        return @sprintf("%.2fs", seconds)
    elseif seconds < 3600
        minutes = floor(Int, seconds / 60)
        return @sprintf("%dm %.2fs", minutes, seconds - 60 * minutes)
    end
    hours = floor(Int, seconds / 3600)
    minutes = floor(Int, (seconds - 3600 * hours) / 60)
    return @sprintf("%dh %dm %.2fs", hours, minutes, seconds - 3600 * hours - 60 * minutes)
end

function log_cache_status(name::AbstractString, hit::Bool, hash::AbstractString, dir::AbstractString)
    status = hit ? "cache hit" : "cache miss"
    log_message("cache", @sprintf("%s: %s (hash=%s, path=%s)", name, status, hash, dir))
    return nothing
end

include(joinpath(@__DIR__, "CoupledL63Config.jl"))
include(joinpath(@__DIR__, "CoupledL63Paths.jl"))
include(joinpath(@__DIR__, "CoupledL63Simulation.jl"))
include(joinpath(@__DIR__, "CoupledL63Score.jl"))
include(joinpath(@__DIR__, "CoupledL63PhiSigma.jl"))
include(joinpath(@__DIR__, "CoupledL63Residuals.jl"))

function parse_args(argv=ARGS)
    args = Dict{String,String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if startswith(arg, "--")
            key = lowercase(arg[3:end])
            if i == length(argv) || startswith(argv[i + 1], "--")
                args[key] = "true"
            else
                args[key] = argv[i + 1]
                i += 1
            end
        end
        i += 1
    end
    return args
end

function config_from_file(config_path::String, args::Dict{String,String})
    raw = TOML.parsefile(config_path)
    parsed = parse_config(raw)
    execution = parsed.execution
    if haskey(args, "stage")
        execution = ExecutionConfig(
            stages=[canonical_stage(args["stage"])],
        )
    end
    return CoupledL63Config(
        execution=execution,
        system=parsed.system,
        high_resolution=parsed.high_resolution,
        low_resolution=parsed.low_resolution,
        score=parsed.score,
        phi_sigma=parsed.phi_sigma,
        rom=parsed.rom,
        residuals=parsed.residuals,
        raw=raw,
    )
end

function save_run_summary(path::String, summary::Dict{String,Any})
    open(path, "w") do io
        TOML.print(io, summary)
    end
    return nothing
end

stage_requested(config::CoupledL63Config, stage::Symbol) = stage in config.execution.stages

function render_figure(renderer_path::String, kind::String, data_dir::String, png_path::String, pdf_path::String, title::String)
    log_message("figure", @sprintf("Rendering %s figure from %s", kind, data_dir))
    run(`python3 $(renderer_path) $(kind) $(data_dir) $(png_path) $(pdf_path) $(title)`)
    log_message("figure", @sprintf("Finished %s figure: %s", kind, png_path))
    return nothing
end

function dataset_cache_exists(dir::String)
    return isfile(dataset_h5_path(dir)) &&
        isfile(cache_manifest_path(dir)) &&
        isfile(cache_parameters_path(dir))
end

function save_dataset_cache_metadata(dir::String,
                                     artifact_type::String,
                                     artifact_hash::String,
                                     cache_params::AbstractDict,
                                     dependencies::AbstractDict)
    manifest = Dict(
        "artifact_type" => artifact_type,
        "artifact_hash" => artifact_hash,
        "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "dependencies" => dependencies,
    )
    write_toml_file(cache_manifest_path(dir), manifest)
    write_toml_file(cache_parameters_path(dir), cache_params)
    return nothing
end

function high_res_cache_payload(system::SystemConfig, high_cfg::HighResolutionConfig)
    return Dict(
        "system" => Dict(
            "label" => system.label,
            "forcing_x1" => String(system.forcing_x1),
            "forcing_x2" => String(system.forcing_x2),
            "eps" => system.eps,
            "kappa" => system.kappa,
            "Omega" => system.Omega,
            "a1" => system.a1,
            "a2" => system.a2,
            "b1" => system.b1,
            "b2" => system.b2,
            "sigma_x1" => system.sigma_x1,
            "sigma_x2" => system.sigma_x2,
            "base_y2_ref" => system.base_y2_ref,
            "base_y3_ref" => system.base_y3_ref,
        ),
        "integration" => Dict(
            "dt" => high_cfg.dt,
            "t_reference" => high_cfg.t_reference,
            "t_reference_transient" => high_cfg.t_reference_transient,
            "t_total" => high_cfg.t_total,
            "t_transient" => high_cfg.t_transient,
            "sample_stride" => high_cfg.sample_stride,
        ),
    )
end

function low_res_cache_payload(system::SystemConfig, low_cfg::LowResolutionConfig, high_hash::String)
    return Dict(
        "system" => Dict(
            "label" => system.label,
            "forcing_x1" => String(system.forcing_x1),
            "forcing_x2" => String(system.forcing_x2),
            "eps" => system.eps,
            "kappa" => system.kappa,
            "Omega" => system.Omega,
            "a1" => system.a1,
            "a2" => system.a2,
            "b1" => system.b1,
            "b2" => system.b2,
            "sigma_x1" => system.sigma_x1,
            "sigma_x2" => system.sigma_x2,
            "base_y2_ref" => system.base_y2_ref,
            "base_y3_ref" => system.base_y3_ref,
        ),
        "low_resolution" => Dict(
            "training_target_uncorrelated" => low_cfg.training_target_uncorrelated,
            "training_seed_stride_multiplier" => low_cfg.training_seed_stride_multiplier,
        ),
        "high_resolution_dataset_hash" => high_hash,
    )
end

function resolve_data_artifacts(config::CoupledL63Config, roots::RootPaths)
    high_payload = high_res_cache_payload(config.system, config.high_resolution)
    high_hash = artifact_hash(high_payload)
    high_dir = high_res_cache_dir(roots, high_hash)

    low_payload = low_res_cache_payload(config.system, config.low_resolution, high_hash)
    low_hash = artifact_hash(low_payload)
    low_dir = low_res_cache_dir(roots, low_hash)

    return (
        high_hash=high_hash,
        high_dir=high_dir,
        high_payload=high_payload,
        low_hash=low_hash,
        low_dir=low_dir,
        low_payload=low_payload,
    )
end

function run_summary_base(run_paths::RunPaths, config_path::String)
    return Dict{String,Any}(
        "run_id" => run_paths.run_id,
        "run_name" => run_paths.run_name,
        "run_dir" => run_paths.run_dir,
        "config_path" => config_path,
        "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "completed_stages" => String[],
        "artifacts" => Dict{String,Any}(),
    )
end

function mark_stage_complete!(summary::Dict{String,Any}, stage::Symbol)
    completed = Set(String.(summary["completed_stages"]))
    push!(completed, String(stage))
    summary["completed_stages"] = [String(stage) for stage in STAGE_SEQUENCE if String(stage) in completed]
    return summary
end

function save_high_resolution_cache(dir::String,
                                    payload::AbstractDict,
                                    system::SystemConfig,
                                    params::ModelParameters,
                                    refs::ReferenceLevels,
                                    data::SimulationData)
    mkpath(dir)
    save_dataset_artifact(
        dataset_h5_path(dir),
        system,
        params,
        refs,
        data;
        dataset_role="high_resolution",
        metadata=Dict(
            "dt" => payload["integration"]["dt"],
            "t_reference" => payload["integration"]["t_reference"],
            "t_reference_transient" => payload["integration"]["t_reference_transient"],
            "t_total" => payload["integration"]["t_total"],
            "t_transient" => payload["integration"]["t_transient"],
            "sample_stride" => payload["integration"]["sample_stride"],
        ),
    )
    save_dataset_cache_metadata(dir, "high_resolution_dataset", basename(dir), payload, Dict{String,Any}())
    return nothing
end

function save_low_resolution_cache(dir::String,
                                   payload::AbstractDict,
                                   system::SystemConfig,
                                   params::ModelParameters,
                                   refs::ReferenceLevels,
                                   data::SimulationData,
                                   high_hash::String,
                                   decor_stride_saved::Int,
                                   decor_taus::Vector{Float64},
                                   seed_count::Int,
                                   samples_per_seed::Int)
    mkpath(dir)
    save_dataset_artifact(
        dataset_h5_path(dir),
        system,
        params,
        refs,
        data;
        dataset_role="low_resolution",
        metadata=Dict(
            "source_high_resolution_hash" => high_hash,
            "decorrelation_stride_saved_samples" => decor_stride_saved,
            "decorrelation_stride_time" => decor_stride_saved * (payload["high_resolution_dataset_hash"] isa AbstractString ? data.sample_dt : data.sample_dt),
            "decorrelation_taus_saved_samples" => decor_taus,
            "seed_count" => seed_count,
            "samples_per_seed" => samples_per_seed,
            "training_target_uncorrelated" => payload["low_resolution"]["training_target_uncorrelated"],
        ),
    )
    save_dataset_cache_metadata(
        dir,
        "low_resolution_dataset",
        basename(dir),
        payload,
        Dict("high_resolution_dataset_hash" => high_hash),
    )
    return nothing
end

function generate_data_stage(config::CoupledL63Config,
                             roots::RootPaths,
                             summary::Dict{String,Any};
                             rerun::Bool=false,
                             mark_complete::Bool=true)
    artifacts = resolve_data_artifacts(config, roots)
    log_message("generate_data", @sprintf("Resolved dataset artifacts: high=%s low=%s", artifacts.high_hash, artifacts.low_hash))

    high_hit = dataset_cache_exists(artifacts.high_dir) && !rerun
    log_cache_status("high-resolution dataset", high_hit, artifacts.high_hash, artifacts.high_dir)
    if high_hit
        log_message("generate_data", "Loading cached high-resolution dataset")
        high_dataset = load_dataset_artifact(dataset_h5_path(artifacts.high_dir))
    else
        log_message("generate_data", "Estimating fast reference levels")
        refs = estimate_reference_levels(config.system, config.high_resolution)
        log_message(
            "generate_data",
            @sprintf("Reference levels estimated: y2_ref=%.6f, y3_ref=%.6f", refs.y2_ref, refs.y3_ref),
        )
        params = build_parameters(config.system, refs)
        log_message("generate_data", "Running high-resolution 5D simulation")
        data = simulate_high_resolution(config.system, params, config.high_resolution)
        log_message("generate_data", @sprintf("Saving high-resolution dataset with %d samples", length(data.t)))
        save_high_resolution_cache(artifacts.high_dir, artifacts.high_payload, config.system, params, refs, data)
        high_dataset = load_dataset_artifact(dataset_h5_path(artifacts.high_dir))
    end

    low_hit = dataset_cache_exists(artifacts.low_dir) && !rerun
    log_cache_status("low-resolution dataset", low_hit, artifacts.low_hash, artifacts.low_dir)
    if !low_hit
        log_message("generate_data", "Building decorrelated low-resolution dataset")
        low_data, decor_stride_saved, decor_taus, seed_count, samples_per_seed = build_low_resolution_dataset(
            high_dataset.data,
            high_dataset.params,
            config.high_resolution,
            config.low_resolution,
            config.system.label,
        )
        log_message(
            "generate_data",
            @sprintf(
                "Low-resolution dataset ready: seeds=%d, samples_per_seed=%d, target=%d",
                seed_count,
                samples_per_seed,
                length(low_data.t),
            ),
        )
        save_low_resolution_cache(
            artifacts.low_dir,
            artifacts.low_payload,
            config.system,
            high_dataset.params,
            high_dataset.refs,
            low_data,
            artifacts.high_hash,
            decor_stride_saved,
            decor_taus,
            seed_count,
            samples_per_seed,
        )
    end
    low_dataset = load_dataset_artifact(dataset_h5_path(artifacts.low_dir))
    log_message(
        "generate_data",
        @sprintf(
            "Datasets ready: high=%d samples, low=%d samples",
            length(high_dataset.data.t),
            length(low_dataset.data.t),
        ),
    )

    summary["artifacts"]["high_resolution_dataset"] = Dict(
        "hash" => artifacts.high_hash,
        "path" => dataset_h5_path(artifacts.high_dir),
        "cache_hit" => high_hit,
        "n_samples" => length(high_dataset.data.t),
        "sample_dt" => high_dataset.data.sample_dt,
    )
    summary["artifacts"]["low_resolution_dataset"] = Dict(
        "hash" => artifacts.low_hash,
        "path" => dataset_h5_path(artifacts.low_dir),
        "cache_hit" => low_hit,
        "n_samples" => length(low_dataset.data.t),
        "sample_dt" => low_dataset.data.sample_dt,
    )
    mark_complete && mark_stage_complete!(summary, :generate_data)
    return high_dataset, low_dataset, artifacts
end

function fit_score_stage(config::CoupledL63Config,
                         roots::RootPaths,
                         run_paths::RunPaths,
                         summary::Dict{String,Any},
                         high_dataset,
                         low_dataset,
                         artifacts;
                         rerun::Bool=false,
                         render_outputs::Bool=true,
                         mark_complete::Bool=true)
    log_message("fit_score", "Preparing observed diagnostics and stationary reference")
    observed_high, observed_low, stationary_reference_full, observed_diag = build_observed_diagnostics(
        high_dataset,
        low_dataset,
        config.high_resolution,
        config.rom,
        roots,
        artifacts.high_hash,
    )
    rng = MersenneTwister(config.score.rng_seed)
    training_samples = random_subset_columns(observed_low, config.score.max_training_samples, rng)
    stationary_reference, _ = time_subsample_columns(stationary_reference_full, config.score.max_stationary_samples)
    log_message(
        "fit_score",
        @sprintf(
            "Training data prepared: training_samples=%d, stationary_reference=%d",
            size(training_samples, 2),
            size(stationary_reference, 2),
        ),
    )

    score_payload = score_cache_payload(config.score, artifacts.low_hash)
    score_hash = artifact_hash(score_payload)
    score_dir = score_cache_dir(roots, score_hash)
    score_hit = score_cache_exists(score_dir) && !rerun
    compatible_cache = nothing
    if !score_hit && !rerun
        compatible_cache = find_compatible_score_cache(roots, config.score, artifacts.low_hash)
        if compatible_cache !== nothing
            score_dir = compatible_cache.dir
            score_hash = compatible_cache.hash
            score_hit = true
            log_message(
                "fit_score",
                @sprintf("Using compatible cached score artifact %s because fit_score was not requested", score_hash),
            )
        end
    end
    log_cache_status("score model", score_hit, score_hash, score_dir)

    if config.score.use_gpu
        ScoreEstimation._cuda_module()
        ScoreEstimation._cuda_functional()
        log_message("fit_score", "GPU score path requested")
    end

    if score_hit
        log_message("fit_score", "Loading cached score model")
        score_model, score_summary = load_cached_score(score_dir)
        cached_candidate = load_cached_score_candidate(score_dir, score_model)
        if cached_candidate === nothing
            if render_outputs
                log_message("fit_score", "Cached model found but score evaluation artifact is missing; rebuilding evaluation")
                selected_noise = Float64(score_summary["noise_scale"])
                run_candidate = evaluate_score_model(
                    selected_noise,
                    training_samples,
                    score_model,
                    stationary_reference,
                    observed_diag,
                    high_dataset.data.sample_dt,
                    config.score,
                    config.rom,
                    rng,
                    config.score.rng_seed + 55_555,
                )
                save_score_evaluation_cache(score_dir, run_candidate)
            else
                log_message("fit_score", "Loaded cached score model without evaluation because fit_score was not requested")
                run_candidate = placeholder_score_candidate(score_model, score_summary)
            end
        else
            log_message("fit_score", "Loaded cached score evaluation artifact")
            run_candidate = cached_candidate
        end
    else
        log_message("fit_score", @sprintf("Training score candidates for %d noise scale(s)", length(config.score.noise_scales)))
        candidates = ScoreCandidate[]
        for (idx, noise_scale) in enumerate(config.score.noise_scales)
            log_message("fit_score", @sprintf("Evaluating score candidate %d/%d", idx, length(config.score.noise_scales)))
            push!(candidates, evaluate_score_candidate(
                noise_scale,
                training_samples,
                stationary_reference,
                observed_diag,
                high_dataset.data.sample_dt,
                config.score,
                config.rom,
                rng,
                config.score.rng_seed + 10_000 * idx,
            ))
        end
        run_candidate = select_best_score_candidate(candidates)
        log_message(
            "fit_score",
            @sprintf("Selected score candidate σ_noise=%.6f with objective %.6f", run_candidate.noise_scale, run_candidate.objective),
        )
        save_score_cache(
            score_dir,
            run_candidate,
            candidates,
            score_payload,
            Dict("low_resolution_dataset_hash" => artifacts.low_hash),
            high_dataset.system_label,
        )
    end

    if render_outputs
        log_message("fit_score", "Writing score plot data")
        write_score_plot_data(
            run_paths.score_plot_data_dir,
            observed_diag,
            observed_high,
            high_dataset.data.t,
            run_candidate,
            config.rom,
            joinpath(run_paths.score_plot_data_dir, "summary_metrics.txt");
            pdf_reference_samples=observed_low,
        )
        render_figure(
            roots.renderer_path,
            "score",
            run_paths.score_plot_data_dir,
            run_paths.score_figure_png,
            run_paths.score_figure_pdf,
            high_dataset.system_label,
        )
        log_message("fit_score", "Writing training diagnostics plot data")
        write_training_plot_data(
            run_paths.score_plot_data_dir,
            run_candidate,
            joinpath(run_paths.score_plot_data_dir, "training_summary_metrics.txt"),
        )
        render_figure(
            roots.renderer_path,
            "training",
            run_paths.score_plot_data_dir,
            run_paths.score_training_figure_png,
            run_paths.score_training_figure_pdf,
            high_dataset.system_label,
        )
        log_message("fit_score", @sprintf("Score stage finished with objective %.6f", run_candidate.objective))
    end

    summary["artifacts"]["score"] = Dict(
        "hash" => score_hash,
        "path" => score_model_bson_path(score_dir),
        "summary_path" => score_summary_path(score_dir),
        "cache_hit" => score_hit,
        "selected_noise_scale" => run_candidate.noise_scale,
        "score_objective" => run_candidate.objective,
        "training_figure_png" => run_paths.score_training_figure_png,
        "training_figure_pdf" => run_paths.score_training_figure_pdf,
    )
    mark_complete && mark_stage_complete!(summary, :fit_score)
    return run_candidate, score_hash, score_dir, observed_high, observed_low, stationary_reference, observed_diag
end

function fit_phi_sigma_stage(config::CoupledL63Config,
                             roots::RootPaths,
                             run_paths::RunPaths,
                             summary::Dict{String,Any},
                             high_dataset,
                             observed_high,
                             observed_low,
                             stationary_reference,
                             observed_diag,
                             score_candidate::ScoreCandidate,
                             score_hash::String,
                             artifacts;
                             rerun::Bool=false,
                             render_outputs::Bool=true,
                             mark_complete::Bool=true)
    rng = MersenneTwister(config.score.rng_seed + 50_000)
    correlated_samples, _ = time_subsample_columns(observed_high, config.phi_sigma.max_correlated_samples)
    log_message("fit_phi_sigma", @sprintf("Prepared %d correlated samples for Phi/Sigma estimation", size(correlated_samples, 2)))
    phi_payload = phi_sigma_cache_payload(config.phi_sigma, score_hash, artifacts.high_hash)
    phi_hash = artifact_hash(phi_payload)
    cache_dir = phi_sigma_cache_dir(roots, phi_hash)
    phi_hit = phi_sigma_cache_exists(cache_dir) && !rerun
    log_cache_status("phi/sigma artifact", phi_hit, phi_hash, cache_dir)

    if phi_hit
        log_message("fit_phi_sigma", "Loading cached Phi/Sigma artifact")
        cached = load_cached_phi_sigma(cache_dir)
        if cached.evaluation === nothing
            if render_outputs
                log_message("fit_phi_sigma", "Cached Phi/Sigma found but ROM evaluation artifact is missing; rebuilding evaluation")
                final_candidate = build_rom_candidate(
                    score_candidate.score_model,
                    score_candidate.noise_scale,
                    cached.phi_raw,
                    cached.sigma_derived,
                    cached.info,
                    stationary_reference,
                    observed_diag,
                    high_dataset.data.sample_dt,
                    config.score,
                    config.rom,
                    config.phi_sigma,
                    rng,
                    config.score.rng_seed + 99_999,
                )
                save_phi_sigma_cache(
                    cache_dir,
                    final_candidate,
                    phi_payload,
                    Dict(
                        "score_hash" => score_hash,
                        "high_resolution_dataset_hash" => artifacts.high_hash,
                    ),
                    high_dataset.system_label,
                )
            else
                log_message("fit_phi_sigma", "Loaded cached Phi/Sigma matrices without ROM evaluation because fit_phi_sigma was not requested")
                final_candidate = placeholder_rom_candidate(score_candidate.score_model, cached)
            end
        else
            log_message("fit_phi_sigma", "Loaded cached ROM evaluation artifact")
            final_candidate = ROMCandidate(
                noise_scale=cached.evaluation.noise_scale,
                score_model=score_candidate.score_model,
                phi_raw=cached.phi_raw,
                phi_rom=cached.phi_rom,
                sigma_derived=cached.sigma_derived,
                sigma_rom=cached.sigma_rom,
                info=cached.info,
                diagnostics=cached.evaluation.diagnostics,
                metrics=cached.evaluation.metrics,
                objective=cached.evaluation.objective,
                trajectory=cached.evaluation.trajectory,
                snapshots=cached.evaluation.snapshots,
                stationary=cached.evaluation.stationary,
            )
        end
    else
        log_message("fit_phi_sigma", "Estimating Phi and Sigma from scratch")
        phi_raw, sigma_derived, info = raw_phi_and_sigma(
            correlated_samples,
            stationary_reference,
            score_candidate.score_model,
            high_dataset.data.sample_dt,
            config.phi_sigma,
            config.score.batch_size,
            config.score.use_gpu,
            rng,
        )
        log_message("fit_phi_sigma", "Running final ROM rollout and diagnostics")
        final_candidate = build_rom_candidate(
            score_candidate.score_model,
            score_candidate.noise_scale,
            phi_raw,
            sigma_derived,
            info,
            stationary_reference,
            observed_diag,
            high_dataset.data.sample_dt,
            config.score,
            config.rom,
            config.phi_sigma,
            rng,
            config.score.rng_seed + 99_999,
        )
        save_phi_sigma_cache(
            cache_dir,
            final_candidate,
            phi_payload,
            Dict(
                "score_hash" => score_hash,
                "high_resolution_dataset_hash" => artifacts.high_hash,
            ),
            high_dataset.system_label,
        )
    end

    if render_outputs
        log_message("fit_phi_sigma", "Writing ROM plot data")
        write_rom_plot_data(
            run_paths.rom_plot_data_dir,
            observed_diag,
            observed_high,
            high_dataset.data.t,
            final_candidate,
            config.rom,
            joinpath(run_paths.rom_plot_data_dir, "summary_metrics.txt");
            pdf_reference_samples=observed_low,
        )
        render_figure(
            roots.renderer_path,
            "rom",
            run_paths.rom_plot_data_dir,
            run_paths.rom_figure_png,
            run_paths.rom_figure_pdf,
            high_dataset.system_label,
        )
        log_message("fit_phi_sigma", @sprintf("Phi/Sigma stage finished with objective %.6f", final_candidate.objective))
    end

    summary["artifacts"]["phi_sigma"] = Dict(
        "hash" => phi_hash,
        "path" => phi_sigma_h5_path(cache_dir),
        "summary_path" => phi_sigma_summary_path(cache_dir),
        "cache_hit" => phi_hit,
        "clusters" => Int(final_candidate.info[:n_states]),
        "objective" => final_candidate.objective,
        "use_identity_phi" => config.phi_sigma.use_identity_phi,
        "use_identity_sigma" => config.phi_sigma.use_identity_sigma,
    )
    mark_complete && mark_stage_complete!(summary, :fit_phi_sigma)
    return final_candidate, phi_hash, cache_dir
end

function estimate_residuals_stage(config::CoupledL63Config,
                                  roots::RootPaths,
                                  run_paths::RunPaths,
                                  summary::Dict{String,Any},
                                  high_dataset,
                                  score_candidate::ScoreCandidate,
                                  rom_candidate::ROMCandidate,
                                  phi_hash::String)
    log_message("residuals", "Recovering residual trajectories and diagnostics")
    result = recover_residuals(
        high_dataset.data,
        high_dataset.params,
        score_candidate.score_model,
        rom_candidate.phi_raw,
        config.residuals,
    )
    log_message("residuals", "Writing residual plot data")
    write_residual_plot_data(
        run_paths.residual_plot_data_dir,
        result,
        config.residuals,
        joinpath(run_paths.residual_plot_data_dir, "summary_metrics.txt"),
    )
    render_figure(
        roots.renderer_path,
        "residual",
        run_paths.residual_plot_data_dir,
        run_paths.residual_figure_png,
        run_paths.residual_figure_pdf,
        high_dataset.system_label,
    )
    log_message(
        "residuals",
        @sprintf("Residual stage finished: r1_rmse=%.6f, r2_rmse=%.6f", result.metrics_r1.rmse, result.metrics_r2.rmse),
    )
    summary["artifacts"]["residuals"] = Dict(
        "source_phi_sigma_hash" => phi_hash,
        "r1_rmse" => result.metrics_r1.rmse,
        "r2_rmse" => result.metrics_r2.rmse,
        "figure_png" => run_paths.residual_figure_png,
        "figure_pdf" => run_paths.residual_figure_pdf,
    )
    mark_stage_complete!(summary, :estimate_residuals)
    return result
end

function run_pipeline(config::CoupledL63Config, config_path::String)
    roots = root_paths(config_path)
    mkpath(roots.data_dir)
    run_paths = allocate_run_paths(roots)
    write_run_parameter_snapshot(run_paths, config.raw)
    summary = run_summary_base(run_paths, config_path)
    log_message("pipeline", @sprintf("Run directory: %s", run_paths.run_dir))

    high_dataset = nothing
    low_dataset = nothing
    artifacts = nothing
    score_candidate = nothing
    score_hash = ""
    observed_high = nothing
    observed_low = nothing
    stationary_reference = nothing
    observed_diag = nothing
    rom_candidate = nothing
    phi_hash = ""
    executed_requested_stages = Set{Symbol}()

    function ensure_generate_data(; as_dependency::Bool=false)
        if high_dataset !== nothing && low_dataset !== nothing && artifacts !== nothing
            return
        end
        rerun = stage_requested(config, :generate_data) && !(:generate_data in executed_requested_stages)
        as_dependency && log_message(
            "pipeline",
            rerun ? "Running requested stage `generate_data` as a prerequisite" : "Loading/building dependency `generate_data`",
        )
        high_dataset, low_dataset, artifacts = generate_data_stage(
            config,
            roots,
            summary;
            rerun=rerun,
            mark_complete=rerun,
        )
        rerun && push!(executed_requested_stages, :generate_data)
    end

    function ensure_score(; as_dependency::Bool=false)
        if score_candidate !== nothing && observed_high !== nothing && observed_low !== nothing && stationary_reference !== nothing && observed_diag !== nothing
            return
        end
        ensure_generate_data(as_dependency=true)
        rerun = stage_requested(config, :fit_score) && !(:fit_score in executed_requested_stages)
        as_dependency && log_message(
            "pipeline",
            rerun ? "Running requested stage `fit_score` as a prerequisite" : "Loading/building dependency `fit_score`",
        )
        score_candidate, score_hash, _, observed_high, observed_low, stationary_reference, observed_diag = fit_score_stage(
            config,
            roots,
            run_paths,
            summary,
            high_dataset,
            low_dataset,
            artifacts;
            rerun=rerun,
            render_outputs=rerun,
            mark_complete=rerun,
        )
        rerun && push!(executed_requested_stages, :fit_score)
    end

    function ensure_phi_sigma(; as_dependency::Bool=false)
        if rom_candidate !== nothing && phi_hash != ""
            return
        end
        ensure_score(as_dependency=true)
        rerun = stage_requested(config, :fit_phi_sigma) && !(:fit_phi_sigma in executed_requested_stages)
        as_dependency && log_message(
            "pipeline",
            rerun ? "Running requested stage `fit_phi_sigma` as a prerequisite" : "Loading/building dependency `fit_phi_sigma`",
        )
        rom_candidate, phi_hash, _ = fit_phi_sigma_stage(
            config,
            roots,
            run_paths,
            summary,
            high_dataset,
            observed_high,
            observed_low,
            stationary_reference,
            observed_diag,
            score_candidate,
            score_hash,
            artifacts;
            rerun=rerun,
            render_outputs=rerun,
            mark_complete=rerun,
        )
        rerun && push!(executed_requested_stages, :fit_phi_sigma)
    end

    for stage in config.execution.stages
        if stage in executed_requested_stages
            log_message("pipeline", @sprintf("Skipping stage `%s` because it already ran as a prerequisite", String(stage)))
            continue
        end
        stage_started = time_ns()
        log_message("pipeline", @sprintf("Starting stage `%s`", String(stage)))
        if stage === :generate_data
            ensure_generate_data()
            push!(executed_requested_stages, :generate_data)
        elseif stage === :fit_score
            ensure_score()
            push!(executed_requested_stages, :fit_score)
        elseif stage === :fit_phi_sigma
            ensure_phi_sigma()
            push!(executed_requested_stages, :fit_phi_sigma)
        elseif stage === :estimate_residuals
            ensure_phi_sigma(as_dependency=true)
            estimate_residuals_stage(
                config,
                roots,
                run_paths,
                summary,
                high_dataset,
                score_candidate,
                rom_candidate,
                phi_hash,
            )
            push!(executed_requested_stages, :estimate_residuals)
        else
            error("Unsupported stage `$(stage)`.")
        end
        save_run_summary(run_paths.summary_toml, summary)
        elapsed = (time_ns() - stage_started) / 1.0e9
        log_message("pipeline", @sprintf("Completed stage `%s` in %s", String(stage), format_elapsed(elapsed)))
    end

    return run_paths, summary
end

function main(argv=ARGS)
    args = parse_args(argv)
    config_path = abspath(get(args, "config", DEFAULT_CONFIG))
    config = config_from_file(config_path, args)
    log_message(
        "pipeline",
        @sprintf(
            "Launching coupled_l63 pipeline with stages=%s, config=%s",
            join(String.(config.execution.stages), ","),
            config_path,
        ),
    )
    run_paths, summary = run_pipeline(config, config_path)
    println("Completed coupled_l63 run:")
    println("  " * run_paths.summary_toml)
    return run_paths, summary
end

end
