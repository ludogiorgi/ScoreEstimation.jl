module BrusselatorPipeline

using CairoMakie
import CUDA
using Dates
using Flux
using ForwardDiff
using HDF5
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using TOML

using ScoreEstimation

CairoMakie.activate!()

const SCRIPT_DIR = @__DIR__
const DEFAULT_CONFIG = joinpath(SCRIPT_DIR, "brusselator_pipeline.toml")
const STAGE_SEQUENCE = (:generate_data, :train_score, :estimate_phi_sigma, :observed_residuals, :overview, :joint_score, :delta_m, :delta_m_rollout, :delta_m_coefficients)
const RAW_LABEL = "Raw DSM"
const METHOD_COLOR = :dodgerblue3
const OBS_COLOR = :black

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

function require_section(config::Dict{String,Any}, key::String)
    haskey(config, key) || error("Missing required config section `$(key)` in $(DEFAULT_CONFIG).")
    return config[key]
end

config_path(args::Dict{String,String}) = abspath(get(args, "config", DEFAULT_CONFIG))

parse_bool(value, default::Bool=false) = value === nothing ? default : parse_bool(value)
parse_bool(value::Bool) = value
function parse_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("1", "true", "yes", "on") && return true
    normalized in ("0", "false", "no", "off") && return false
    error("Unable to parse Bool from `$(value)`.")
end

function resolve_divergence_mode(value)
    mode = Symbol(lowercase(strip(String(value))))
    mode in (:autodiff, :finite_difference) || error("Unsupported divergence mode `$(value)`.")
    return mode
end

canonical_stage(stage::Symbol) = stage
function canonical_stage(stage)
    normalized = lowercase(strip(String(stage)))
    stage_symbol = Symbol(normalized)
    stage_symbol in STAGE_SEQUENCE || error("Unsupported Brusselator stage `$(stage)`.")
    return stage_symbol
end

function resolve_lag_max_step(data, override_time)
    if override_time !== nothing
        tmax = Float64(override_time)
        tmax > 0 || error("Lag-time override must be positive.")
        return clamp(round(Int, tmax / data.sample_dt), 1, size(data.correlated, 2) - 1)
    end

    if data.decorrelation_stride > 0
        return data.decorrelation_stride
    end

    inferred = isempty(data.decorrelation_times) ? 1.0 : maximum(data.decorrelation_times)
    return max(1, ceil(Int, inferred))
end

function requested_stages(config::Dict{String,Any}, args::Dict{String,String}, default_stage::Union{Nothing,Symbol}=nothing)
    if default_stage !== nothing
        return [default_stage]
    end
    if haskey(args, "stage")
        return [canonical_stage(args["stage"])]
    end
    if haskey(args, "stages")
        return [canonical_stage(stage) for stage in split(args["stages"], ',') if !isempty(strip(stage))]
    end

    configured = get(config["run"], "stages", String.(STAGE_SEQUENCE))
    if configured isa AbstractVector
        return [canonical_stage(stage) for stage in configured]
    end
    return [canonical_stage(configured)]
end

function validate_config(config::Dict{String,Any})
    run_cfg = require_section(config, "run")
    paths_cfg = require_section(config, "paths")
    model_cfg = require_section(config, "model")
    generation_cfg = require_section(config, "data_generation")
    training_cfg = require_section(config, "training")
    estimator_cfg = require_section(config, "phi_sigma")
    integration_cfg = require_section(config, "integration")
    diagnostics_cfg = require_section(config, "diagnostics")
    residual_cfg = require_section(config, "observed_residuals")
    joint_score_cfg = require_section(config, "joint_score")
    delta_m_cfg = require_section(config, "delta_m")
    delta_m_rollout_cfg = require_section(config, "delta_m_rollout")
    delta_m_coeff_cfg = require_section(config, "delta_m_coefficients")

    requested_stages(config, Dict{String,String}())

    for key in ("name",)
        haskey(run_cfg, key) || error("Missing `run.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("runs_dir", "dataset_h5", "score_model_bson", "training_h5", "phi_sigma_h5", "diagnostics_h5", "overview_png", "overview_pdf", "observed_residuals_h5", "observed_residuals_png", "observed_residuals_pdf", "joint_score_model_bson", "joint_score_h5", "joint_score_png", "joint_score_pdf", "delta_m_model_bson", "delta_m_h5", "delta_m_png", "delta_m_pdf", "delta_m_rollout_h5", "delta_m_rollout_png", "delta_m_rollout_pdf", "delta_m_coefficients_h5", "delta_m_coefficients_png", "delta_m_coefficients_pdf", "config_snapshot", "summary_toml", "latest_run_marker")
        haskey(paths_cfg, key) || error("Missing `paths.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("A", "B", "Omega")
        haskey(model_cfg, key) || error("Missing `model.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("dt", "burnin_steps", "n_steps_timeseries", "resolution_timeseries", "n_steps_stationary", "burnin_stationary", "resolution_stationary", "ensemble_stationary", "seed", "verbose")
        haskey(generation_cfg, key) || error("Missing `data_generation.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("seed", "noise_scales", "epochs_schedule", "neurons", "batch_size", "max_batches_per_epoch", "learning_rate", "use_gpu", "gpu_id", "verbose", "mean_regularization", "stein_regularization")
        haskey(training_cfg, key) || error("Missing `training.$(key)` in $(DEFAULT_CONFIG).")
    end
    length(training_cfg["noise_scales"]) == length(training_cfg["epochs_schedule"]) || error("`training.noise_scales` and `training.epochs_schedule` must have the same length.")

    for key in ("seed", "q_min_prob", "batch_size", "regularization", "use_gpu", "use_stein_correction", "enforce_psd_symmetric", "finite_dt_correction", "finite_dt_correction_mode", "generator_perturb_scale", "perturb_scale_override")
        haskey(estimator_cfg, key) || error("Missing `phi_sigma.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("dt", "n_steps", "burn_in", "resolution", "n_ensembles", "use_gpu", "progress", "seed")
        haskey(integration_cfg, key) || error("Missing `integration.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("maxlag", "plot_window", "phase_points", "metric_time_horizon")
        haskey(diagnostics_cfg, key) || error("Missing `diagnostics.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("batch_size", "use_gpu", "include_zero_lag", "lag_step_stride", "max_lag_fraction")
        haskey(residual_cfg, key) || error("Missing `observed_residuals.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("seed", "noise_scales", "epochs_schedule", "neurons", "batch_size", "max_batches_per_epoch", "learning_rate", "use_gpu", "gpu_id", "verbose", "mean_regularization", "stein_regularization", "lag_fractions", "eval_batches_per_lag", "sampler_dt", "sampler_steps", "sampler_burn_in", "sampler_resolution", "sampler_ensembles", "sampler_use_gpu", "sampler_progress", "sampler_boundary", "joint_pdf_bins", "contour_levels")
        haskey(joint_score_cfg, key) || error("Missing `joint_score.$(key)` in $(DEFAULT_CONFIG).")
    end
    length(joint_score_cfg["noise_scales"]) == length(joint_score_cfg["epochs_schedule"]) || error("`joint_score.noise_scales` and `joint_score.epochs_schedule` must have the same length.")

    for key in ("seed", "tD_override", "lag_step_stride", "epochs", "batch_size", "max_batches_per_epoch", "learning_rate", "neurons", "eps_floor", "mean_regularization", "magnitude_regularization", "use_gpu", "gpu_id", "verbose", "eval_pairs_per_lag", "grid_nx", "grid_ny", "grid_margin_fraction", "check_samples")
        haskey(delta_m_cfg, key) || error("Missing `delta_m.$(key)` in $(DEFAULT_CONFIG).")
    end

    for key in ("dt", "n_steps", "burn_in", "resolution", "n_ensembles", "seed", "progress", "grid_nx", "grid_ny", "grid_margin_fraction", "metric_time_horizon")
        haskey(delta_m_rollout_cfg, key) || error("Missing `delta_m_rollout.$(key)` in $(DEFAULT_CONFIG).")
    end
    for key in ("grid_nx", "grid_ny", "padding_fraction", "lower_quantile", "upper_quantile")
        haskey(delta_m_coeff_cfg, key) || error("Missing `delta_m_coefficients.$(key)` in $(DEFAULT_CONFIG).")
    end

    return config
end

function run_name(config::Dict{String,Any}, args::Dict{String,String})
    return get(args, "run-name", get(config["run"], "name", "default"))
end

function artifact_paths(config::Dict{String,Any}, args::Dict{String,String})
    paths_cfg = config["paths"]
    runs_root = abspath(joinpath(SCRIPT_DIR, get(paths_cfg, "runs_dir", "runs")))
    run_label = run_name(config, args)
    run_dir = joinpath(runs_root, run_label)

    function run_path(key::String, fallback::String)
        return abspath(joinpath(run_dir, get(paths_cfg, key, fallback)))
    end

    return (
        run_name=run_label,
        runs_root=runs_root,
        run_dir=run_dir,
        dataset_h5=run_path("dataset_h5", "data/dataset.h5"),
        score_model_bson=run_path("score_model_bson", "models/score_model.bson"),
        training_h5=run_path("training_h5", "metrics/training.h5"),
        phi_sigma_h5=run_path("phi_sigma_h5", "metrics/phi_sigma.h5"),
        diagnostics_h5=run_path("diagnostics_h5", "metrics/diagnostics.h5"),
        overview_png=run_path("overview_png", "figures/overview.png"),
        overview_pdf=run_path("overview_pdf", "figures/overview.pdf"),
        observed_residuals_h5=run_path("observed_residuals_h5", "metrics/observed_residuals.h5"),
        observed_residuals_png=run_path("observed_residuals_png", "figures/observed_residuals.png"),
        observed_residuals_pdf=run_path("observed_residuals_pdf", "figures/observed_residuals.pdf"),
        joint_score_model_bson=run_path("joint_score_model_bson", "models/joint_score_model.bson"),
        joint_score_h5=run_path("joint_score_h5", "metrics/joint_score.h5"),
        joint_score_png=run_path("joint_score_png", "figures/joint_score.png"),
        joint_score_pdf=run_path("joint_score_pdf", "figures/joint_score.pdf"),
        delta_m_model_bson=run_path("delta_m_model_bson", "models/delta_m_model.bson"),
        delta_m_h5=run_path("delta_m_h5", "metrics/delta_m.h5"),
        delta_m_png=run_path("delta_m_png", "figures/delta_m.png"),
        delta_m_pdf=run_path("delta_m_pdf", "figures/delta_m.pdf"),
        delta_m_rollout_h5=run_path("delta_m_rollout_h5", "metrics/delta_m_rollout.h5"),
        delta_m_rollout_png=run_path("delta_m_rollout_png", "figures/delta_m_rollout.png"),
        delta_m_rollout_pdf=run_path("delta_m_rollout_pdf", "figures/delta_m_rollout.pdf"),
        delta_m_coefficients_h5=run_path("delta_m_coefficients_h5", "metrics/delta_m_coefficients.h5"),
        delta_m_coefficients_png=run_path("delta_m_coefficients_png", "figures/delta_m_coefficients.png"),
        delta_m_coefficients_pdf=run_path("delta_m_coefficients_pdf", "figures/delta_m_coefficients.pdf"),
        config_snapshot=run_path("config_snapshot", "config_snapshot.toml"),
        summary_toml=run_path("summary_toml", "run_summary.toml"),
        latest_run_marker=abspath(joinpath(SCRIPT_DIR, get(paths_cfg, "latest_run_marker", "latest_run.txt"))),
    )
end

function ensure_parent_dirs(paths)
    for path in (
        paths.dataset_h5,
        paths.score_model_bson,
        paths.training_h5,
        paths.phi_sigma_h5,
        paths.diagnostics_h5,
        paths.overview_png,
        paths.overview_pdf,
        paths.observed_residuals_h5,
        paths.observed_residuals_png,
        paths.observed_residuals_pdf,
        paths.joint_score_model_bson,
        paths.joint_score_h5,
        paths.joint_score_png,
        paths.joint_score_pdf,
        paths.delta_m_model_bson,
        paths.delta_m_h5,
        paths.delta_m_png,
        paths.delta_m_pdf,
        paths.delta_m_rollout_h5,
        paths.delta_m_rollout_png,
        paths.delta_m_rollout_pdf,
        paths.delta_m_coefficients_h5,
        paths.delta_m_coefficients_png,
        paths.delta_m_coefficients_pdf,
        paths.config_snapshot,
        paths.summary_toml,
        paths.latest_run_marker,
    )
        mkpath(dirname(path))
    end
    mkpath(paths.run_dir)
    return nothing
end

function save_config_snapshot(config::Dict{String,Any}, paths)
    open(paths.config_snapshot, "w") do io
        TOML.print(io, config)
    end
    return nothing
end

function write_latest_run_marker(paths)
    open(paths.latest_run_marker, "w") do io
        println(io, paths.run_dir)
    end
    return nothing
end

function artifact_map(paths)
    return Dict(
        "run_dir" => paths.run_dir,
        "dataset_h5" => paths.dataset_h5,
        "score_model_bson" => paths.score_model_bson,
        "training_h5" => paths.training_h5,
        "phi_sigma_h5" => paths.phi_sigma_h5,
        "diagnostics_h5" => paths.diagnostics_h5,
        "overview_png" => paths.overview_png,
        "overview_pdf" => paths.overview_pdf,
        "observed_residuals_h5" => paths.observed_residuals_h5,
        "observed_residuals_png" => paths.observed_residuals_png,
        "observed_residuals_pdf" => paths.observed_residuals_pdf,
        "joint_score_model_bson" => paths.joint_score_model_bson,
        "joint_score_h5" => paths.joint_score_h5,
        "joint_score_png" => paths.joint_score_png,
        "joint_score_pdf" => paths.joint_score_pdf,
        "delta_m_model_bson" => paths.delta_m_model_bson,
        "delta_m_h5" => paths.delta_m_h5,
        "delta_m_png" => paths.delta_m_png,
        "delta_m_pdf" => paths.delta_m_pdf,
        "delta_m_rollout_h5" => paths.delta_m_rollout_h5,
        "delta_m_rollout_png" => paths.delta_m_rollout_png,
        "delta_m_rollout_pdf" => paths.delta_m_rollout_pdf,
        "delta_m_coefficients_h5" => paths.delta_m_coefficients_h5,
        "delta_m_coefficients_png" => paths.delta_m_coefficients_png,
        "delta_m_coefficients_pdf" => paths.delta_m_coefficients_pdf,
        "config_snapshot" => paths.config_snapshot,
    )
end

function load_summary(paths)
    if isfile(paths.summary_toml)
        return TOML.parsefile(paths.summary_toml)
    end
    return Dict{String,Any}()
end

function write_summary(paths, summary::Dict{String,Any})
    open(paths.summary_toml, "w") do io
        TOML.print(io, summary)
    end
    return nothing
end

function initialize_run!(config::Dict{String,Any}, paths, source_config::AbstractString)
    ensure_parent_dirs(paths)
    save_config_snapshot(config, paths)
    write_latest_run_marker(paths)

    summary = load_summary(paths)
    summary["run"] = Dict(
        "name" => paths.run_name,
        "created_at" => string(now()),
        "source_config" => source_config,
    )
    summary["artifacts"] = artifact_map(paths)
    summary["stages"] = get(summary, "stages", Dict{String,Any}())
    write_summary(paths, summary)
    return nothing
end

function update_stage_summary!(paths, stage::Symbol, info::Dict{String,Any})
    summary = load_summary(paths)
    stages = haskey(summary, "stages") ? summary["stages"] : Dict{String,Any}()
    stages[String(stage)] = info
    summary["stages"] = stages
    write_summary(paths, summary)
    return nothing
end

function update_results_summary!(paths, results::AbstractDict{String,<:Any})
    summary = load_summary(paths)
    summary["results"] = Dict{String,Any}(results)
    write_summary(paths, summary)
    return nothing
end

function stage_elapsed(stage_fn)
    t0 = time()
    result = stage_fn()
    return result, time() - t0
end

function require_artifact(path::AbstractString, producer::Symbol, consumer::Symbol)
    isfile(path) || error("Missing required artifact for `$(consumer)`: $(path). Run `$(producer)` first or use `--stages`/`--force` to regenerate it.")
    return path
end

function build_generation_config(config::Dict{String,Any})
    model = config["model"]
    generation = config["data_generation"]
    diagnostics = config["diagnostics"]
    return BrusselatorConfig(
        A=Float64(model["A"]),
        B=Float64(model["B"]),
        Ω=Float64(model["Omega"]),
        dt=Float64(generation["dt"]),
        burnin_steps=Int(generation["burnin_steps"]),
        n_steps_timeseries=Int(generation["n_steps_timeseries"]),
        resolution_timeseries=Int(generation["resolution_timeseries"]),
        n_steps_stationary=Int(generation["n_steps_stationary"]),
        burnin_stationary=Int(generation["burnin_stationary"]),
        resolution_stationary=Int(generation["resolution_stationary"]),
        ensemble_stationary=Int(generation["ensemble_stationary"]),
        maxlag=Int(diagnostics["maxlag"]),
        plot_window=Int(diagnostics["plot_window"]),
        phase_points=Int(diagnostics["phase_points"]),
        seed=Int(generation["seed"]),
        verbose=parse_bool(get(generation, "verbose", true)),
    )
end

function build_eval_config(base_cfg::BrusselatorConfig, config::Dict{String,Any})
    diagnostics = config["diagnostics"]
    return BrusselatorConfig(
        A=base_cfg.A,
        B=base_cfg.B,
        Ω=base_cfg.Ω,
        dt=base_cfg.dt,
        burnin_steps=base_cfg.burnin_steps,
        n_steps_timeseries=base_cfg.n_steps_timeseries,
        resolution_timeseries=base_cfg.resolution_timeseries,
        n_steps_stationary=base_cfg.n_steps_stationary,
        burnin_stationary=base_cfg.burnin_stationary,
        resolution_stationary=base_cfg.resolution_stationary,
        ensemble_stationary=base_cfg.ensemble_stationary,
        maxlag=Int(diagnostics["maxlag"]),
        plot_window=Int(diagnostics["plot_window"]),
        phase_points=Int(diagnostics["phase_points"]),
        seed=base_cfg.seed,
        verbose=false,
    )
end

function available_gpu_ids()
    CUDA.functional() || return Int[]
    return collect(eachindex(collect(CUDA.devices())))
end

function set_gpu!(gpu_id::Int)
    CUDA.functional() || error("CUDA is unavailable.")
    devices = collect(CUDA.devices())
    gpu_id in eachindex(devices) || error("GPU id $(gpu_id) is unavailable. Found $(length(devices)) CUDA devices.")
    CUDA.device!(devices[gpu_id])
    return nothing
end

function normalize_dataset(samples::AbstractMatrix)
    μ = vec(mean(samples; dims=2))
    σ = vec(std(samples; dims=2))
    σ .= max.(σ, 1.0f-6)
    normalized = (samples .- μ) ./ σ
    return normalized, μ, σ
end

function sample_initial_conditions(stationary::AbstractMatrix, n_ensembles::Int, seed::Int)
    rng = MersenneTwister(seed)
    x0 = Matrix{Float32}(undef, size(stationary, 1), n_ensembles)
    for ensemble_idx in 1:n_ensembles
        sample_idx = rand(rng, axes(stationary, 2))
        @views x0[:, ensemble_idx] .= stationary[:, sample_idx]
    end
    return x0
end

stationary_from_snapshots(traj::Array{Float32,3}) = reshape(traj, size(traj, 1), :)
first_trajectory(traj::Array{Float32,3}) = Array(traj[:, :, 1])

function load_dataset(path::AbstractString)
    h5open(path, "r") do file
        cfg = BrusselatorConfig(
            A=read(file["A"]),
            B=read(file["B"]),
            Ω=read(file["Omega"]),
            dt=read(file["dt"]),
            burnin_steps=Int(read(file["burnin_steps"])),
            n_steps_timeseries=Int(read(file["n_steps_timeseries"])),
            resolution_timeseries=Int(read(file["resolution_timeseries"])),
            n_steps_stationary=Int(read(file["n_steps_stationary"])),
            burnin_stationary=Int(read(file["burnin_stationary"])),
            resolution_stationary=Int(read(file["resolution_stationary"])),
            ensemble_stationary=Int(read(file["ensemble_stationary"])),
            maxlag=Int(read(file["maxlag"])),
            plot_window=Int(read(file["plot_window"])),
            phase_points=Int(read(file["phase_points"])),
            seed=Int(read(file["seed"])),
            verbose=false,
        )
        return (
            cfg=cfg,
            correlated=Float32.(read(file["correlated_samples"])),
            uncorrelated=Float32.(read(file["uncorrelated_samples"])),
            time_axis=Float64.(read(file["time_axis"])),
            sample_dt=Float64(read(file["dt"])) * Int(read(file["resolution_timeseries"])),
            decorrelation_times=haskey(file, "decorrelation_times") ? Float64.(read(file["decorrelation_times"])) : Float64[],
            decorrelation_stride=haskey(file, "decorrelation_stride") ? Int(read(file["decorrelation_stride"])) : 0,
        )
    end
end

function build_score_wrapper(path::AbstractString)
    model, metadata = load_model(path)
    wrapper = ScoreModel(
        model,
        Float64(metadata["noise_scale"]);
        mean=metadata["mean"],
        scale=metadata["scale"],
    )
    return wrapper, metadata
end

function write_sparse_matrix!(group, name::AbstractString, matrix::SparseMatrixCSC)
    sparse_group = create_group(group, name)
    sparse_group["m"] = size(matrix, 1)
    sparse_group["n"] = size(matrix, 2)
    sparse_group["colptr"] = Int.(matrix.colptr)
    sparse_group["rowval"] = Int.(matrix.rowval)
    sparse_group["values"] = Float64.(matrix.nzval)
    return nothing
end

function read_sparse_matrix(group, name::AbstractString)
    sparse_group = group[name]
    m = Int(read(sparse_group["m"]))
    n = Int(read(sparse_group["n"]))
    colptr = Int.(read(sparse_group["colptr"]))
    rowval = Int.(read(sparse_group["rowval"]))
    values = Float64.(read(sparse_group["values"]))
    return SparseMatrixCSC(m, n, colptr, rowval, values)
end

function draw_matrix_panel!(ax, matrix::AbstractMatrix, title::AbstractString)
    bound = max(maximum(abs, matrix), 1.0e-8)
    heatmap!(ax, 1:size(matrix, 2), 1:size(matrix, 1), matrix; colormap=:balance, colorrange=(-bound, bound))
    for i in axes(matrix, 1), j in axes(matrix, 2)
        text!(ax, j, i; text=string(round(matrix[i, j], digits=3)), align=(:center, :center), color=:white, fontsize=18)
    end
    ax.title = title
    return nothing
end

struct ObservableSpec
    name::String
    output_labels::Vector{String}
    group::Symbol
    eval_batch::Function
end

function build_base_observable_specs(samples::AbstractMatrix)
    μ = vec(mean(samples; dims=2))
    μ1, μ2 = Float32.(μ)
    return [
        ObservableSpec(
            "Coordinates",
            ["x_1", "x_2"],
            :coordinate,
            x -> Float32.(x),
        ),
        ObservableSpec(
            "Centered Quadratics",
            ["(x_1-μ_1)^2", "(x_1-μ_1)(x_2-μ_2)", "(x_2-μ_2)^2"],
            :polynomial,
            x -> begin
                c1 = Float32.(@view x[1, :]) .- μ1
                c2 = Float32.(@view x[2, :]) .- μ2
                return vcat(
                    reshape(c1 .^ 2, 1, :),
                    reshape(c1 .* c2, 1, :),
                    reshape(c2 .^ 2, 1, :),
                )
            end,
        ),
    ]
end

function build_delta_m_observable_specs(samples::AbstractMatrix,
                                        cfg::Dict{String,Any}=Dict{String,Any}())
    observables = ObservableSpec[]
    append!(observables, build_base_observable_specs(samples))

    μ = vec(mean(samples; dims=2))
    μ1, μ2 = Float32.(μ)

    if parse_bool(get(cfg, "include_cubics", true))
        push!(observables, ObservableSpec(
            "Centered Cubics",
            ["(x_1-μ_1)^3", "(x_1-μ_1)^2(x_2-μ_2)", "(x_1-μ_1)(x_2-μ_2)^2", "(x_2-μ_2)^3"],
            :polynomial,
            x -> begin
                c1 = Float32.(@view x[1, :]) .- μ1
                c2 = Float32.(@view x[2, :]) .- μ2
                return vcat(
                    reshape(c1 .^ 3, 1, :),
                    reshape((c1 .^ 2) .* c2, 1, :),
                    reshape(c1 .* (c2 .^ 2), 1, :),
                    reshape(c2 .^ 3, 1, :),
                )
            end,
        ))
    end

    if parse_bool(get(cfg, "include_quartics", true))
        push!(observables, ObservableSpec(
            "Centered Quartics",
            ["(x_1-μ_1)^4", "(x_1-μ_1)^3(x_2-μ_2)", "(x_1-μ_1)^2(x_2-μ_2)^2", "(x_1-μ_1)(x_2-μ_2)^3", "(x_2-μ_2)^4"],
            :polynomial,
            x -> begin
                c1 = Float32.(@view x[1, :]) .- μ1
                c2 = Float32.(@view x[2, :]) .- μ2
                return vcat(
                    reshape(c1 .^ 4, 1, :),
                    reshape((c1 .^ 3) .* c2, 1, :),
                    reshape((c1 .^ 2) .* (c2 .^ 2), 1, :),
                    reshape(c1 .* (c2 .^ 3), 1, :),
                    reshape(c2 .^ 4, 1, :),
                )
            end,
        ))
    end

    if parse_bool(get(cfg, "include_rbfs", true))
        nx = Int(get(cfg, "rbf_grid_nx", 3))
        ny = Int(get(cfg, "rbf_grid_ny", 3))
        axes_info = percentile_field_axes(
            samples,
            nx,
            ny;
            lower_quantile=Float64(get(cfg, "rbf_lower_quantile", 0.05)),
            upper_quantile=Float64(get(cfg, "rbf_upper_quantile", 0.95)),
            padding_fraction=Float64(get(cfg, "rbf_padding_fraction", 0.02)),
        )
        xcenters = Float32.(axes_info.x)
        ycenters = Float32.(axes_info.y)
        σx = Float32(get(cfg, "rbf_width_scale", 1.0) * max(length(xcenters) > 1 ? mean(diff(xcenters)) : std(Float32.(samples[1, :])), 1.0f-3))
        σy = Float32(get(cfg, "rbf_width_scale", 1.0) * max(length(ycenters) > 1 ? mean(diff(ycenters)) : std(Float32.(samples[2, :])), 1.0f-3))
        labels = String[]
        for (iy, cy) in enumerate(ycenters), (ix, cx) in enumerate(xcenters)
            push!(labels, "rbf($(ix),$(iy))")
        end
        push!(observables, ObservableSpec(
            "RBF Probes",
            labels,
            :rbf,
            x -> begin
                x1 = Float32.(@view x[1, :])
                x2 = Float32.(@view x[2, :])
                values = similar(x, Float32, length(labels), size(x, 2))
                row = 1
                for cy in ycenters, cx in xcenters
                    @views values[row, :] .= exp.(-0.5f0 .* (((x1 .- cx) ./ σx) .^ 2 .+ ((x2 .- cy) ./ σy) .^ 2))
                    row += 1
                end
                return values
            end,
        ))
    end

    return observables
end

function evaluate_observable(obs::ObservableSpec, samples::AbstractMatrix)
    values = obs.eval_batch(samples)
    return Matrix{Float32}(values)
end

function slugify(name::AbstractString)
    lowered = lowercase(strip(name))
    slug = replace(lowered, r"[^a-z0-9]+" => "_")
    slug = replace(slug, r"(^_+|_+$)" => "")
    isempty(slug) && return "observable"
    return slug
end

function batched_model_outputs(model,
                               samples::AbstractMatrix;
                               batch_size::Integer=8192,
                               use_gpu::Bool=false)
    nsamples = size(samples, 2)
    nsamples > 0 || error("At least one sample is required for model evaluation.")

    model_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(model), CUDA)
    else
        deepcopy(model)
    end
    Flux.testmode!(model_eval)

    first_stop = min(Int(batch_size), nsamples)
    first_batch = Float32.(samples[:, 1:first_stop])
    first_pred = Array(model_eval(ScoreEstimation._to_device(first_batch; use_gpu=use_gpu)))
    outputs = Matrix{Float32}(undef, size(first_pred, 1), nsamples)
    @views outputs[:, 1:first_stop] .= first_pred

    start_idx = first_stop + 1
    while start_idx <= nsamples
        stop_idx = min(start_idx + Int(batch_size) - 1, nsamples)
        batch = Float32.(samples[:, start_idx:stop_idx])
        pred = Array(model_eval(ScoreEstimation._to_device(batch; use_gpu=use_gpu)))
        @views outputs[:, start_idx:stop_idx] .= pred
        start_idx = stop_idx + 1
    end

    return outputs
end

function norm_summary(vectors::AbstractMatrix)
    norms = vec(sqrt.(sum(abs2, Float64.(vectors); dims=1)))
    isempty(norms) && return (mean=0.0, std=0.0, q95=0.0, q99=0.0, max=0.0)
    return (
        mean=mean(norms),
        std=std(norms),
        q95=quantile(norms, 0.95),
        q99=quantile(norms, 0.99),
        max=maximum(norms),
    )
end

function matrix_norm_summary(values::AbstractArray{<:Real,3})
    norms = [norm(Float64.(@view values[:, :, idx])) for idx in axes(values, 3)]
    isempty(norms) && return (mean=0.0, max=0.0)
    return (mean=mean(norms), max=maximum(norms))
end

function local_polyfit_smooth_and_derivative(times::AbstractVector{<:Real},
                                             values::AbstractVector{<:Real};
                                             window_radius::Integer=3,
                                             degree::Integer=2)
    n = length(times)
    n == length(values) || error("times and values must have the same length.")
    n > 0 || error("At least one time sample is required.")
    degree_eff = min(Int(degree), max(n - 1, 0))
    min_points = max(degree_eff + 1, 2)
    smoothed = Vector{Float64}(undef, n)
    derivative = Vector{Float64}(undef, n)
    t = Float64.(times)
    y = Float64.(values)

    for idx in 1:n
        lo = max(1, idx - window_radius)
        hi = min(n, idx + window_radius)
        while (hi - lo + 1) < min_points
            if lo > 1
                lo -= 1
            elseif hi < n
                hi += 1
            else
                break
            end
        end
        inds = lo:hi
        τ = t[inds] .- t[idx]
        A = hcat([τ .^ p for p in 0:degree_eff]...)
        coeffs = A \ y[inds]
        smoothed[idx] = coeffs[1]
        derivative[idx] = degree_eff >= 1 ? coeffs[2] : 0.0
    end

    return smoothed, derivative
end

function smooth_tensor_time_derivative(values::AbstractArray{<:Real,3},
                                       times::AbstractVector{<:Real};
                                       window_radius::Integer=3,
                                       degree::Integer=2)
    size(values, 3) == length(times) || error("time dimension does not match lag times.")
    smoothed = Array{Float64}(undef, size(values))
    derivative = Array{Float64}(undef, size(values))
    for i in axes(values, 1), j in axes(values, 2)
        curve = Float64.(vec(@view values[i, j, :]))
        smooth_curve, deriv_curve = local_polyfit_smooth_and_derivative(
            times,
            curve;
            window_radius=window_radius,
            degree=degree,
        )
        @views smoothed[i, j, :] .= smooth_curve
        @views derivative[i, j, :] .= deriv_curve
    end
    return smoothed, derivative
end

function relative_frobenius_error_by_slice(reference::AbstractArray{<:Real,3},
                                           candidate::AbstractArray{<:Real,3})
    size(reference) == size(candidate) || error("reference and candidate must have the same shape.")
    errors = Vector{Float64}(undef, size(reference, 3))
    for idx in axes(reference, 3)
        ref = Float64.(@view reference[:, :, idx])
        cand = Float64.(@view candidate[:, :, idx])
        errors[idx] = norm(ref .- cand) / max(norm(ref), eps(Float64))
    end
    return errors
end

function generator_state_derivative_series(Q::SparseMatrixCSC,
                                           centers::AbstractMatrix,
                                           pi_vec::AbstractVector,
                                           lag_steps::AbstractVector{<:Integer},
                                           sample_dt::Real)
    lag_steps_sorted = sort(unique(Int.(lag_steps)))
    X = Float64.(centers)
    pi_norm = Float64.(pi_vec)
    pi_norm ./= sum(pi_norm)
    propagator = exp(Float64(sample_dt) * Matrix(Q))
    state = Diagonal(pi_norm) * transpose(X)
    n_states = size(Q, 1)
    dim = size(X, 1)
    series = Array{Float64}(undef, n_states, dim, length(lag_steps_sorted))
    current_lag = 0

    for (idx, lag_step) in enumerate(lag_steps_sorted)
        while current_lag < lag_step
            state = propagator * state
            current_lag += 1
        end
        @views series[:, :, idx] .= Q * state
    end

    return lag_steps_sorted, series
end

function empirical_score_observable_tensor(obs::ObservableSpec,
                                           correlated::AbstractMatrix,
                                           score_values::AbstractMatrix,
                                           lag_steps::AbstractVector{<:Integer})
    n_time = size(correlated, 2)
    dim = size(score_values, 1)
    output_dim = length(obs.output_labels)
    values = Array{Float64}(undef, output_dim, dim, length(lag_steps))

    for (idx, lag_step) in enumerate(lag_steps)
        lag = Int(lag_step)
        n_pairs = n_time - lag
        n_pairs > 0 || error("Lag $(lag) leaves no observed pairs.")
        xt = @view correlated[:, (1 + lag):(lag + n_pairs)]
        phi_xt = Float64.(evaluate_observable(obs, xt))
        score_x0 = Float64.(@view score_values[:, 1:n_pairs])
        @views values[:, :, idx] .= (phi_xt * transpose(score_x0)) ./ n_pairs
    end

    return values
end

function observable_residual_figure(observable_results, lag_times::AbstractVector{<:Real})
    total_output_rows = sum(length(result.output_labels) + 1 for result in observable_results)
    fig = Figure(size=(2200, 260 * total_output_rows))
    grid_row = 1

    for result in observable_results
        Label(fig[grid_row, 1:3], result.name, fontsize=26, font=:bold)
        grid_row += 1
        output_dim = length(result.output_labels)
        for output_idx in 1:output_dim
            for state_idx in 1:2
                primary = hasproperty(result, :Ephi_Q) ? result.Ephi_Q : result.Ephi
                secondary = hasproperty(result, :Ephi_emp) ? result.Ephi_emp : nothing
                ax = Axis(
                    fig[grid_row + output_idx - 1, state_idx],
                    xlabel=output_idx == output_dim ? "t" : "",
                    ylabel=state_idx == 1 ? result.output_labels[output_idx] : "",
                    title="E[$(output_idx),$(state_idx)]",
                )
                lines!(ax, lag_times, vec(primary[output_idx, state_idx, :]); color=METHOD_COLOR, linewidth=2.5)
                secondary === nothing || lines!(ax, lag_times, vec(secondary[output_idx, state_idx, :]); color=:darkorange2, linewidth=2.0, linestyle=:dash)
                hlines!(ax, [0.0]; color=:gray60, linestyle=:dash)
            end
        end

        norm_ax = Axis(
            fig[grid_row:(grid_row + output_dim - 1), 3],
            xlabel="t",
            ylabel="||E||_F",
            title="$(result.name) residual norm",
        )
        primary_norm = hasproperty(result, :fro_norm_q) ? result.fro_norm_q : result.fro_norm
        secondary_norm = hasproperty(result, :fro_norm_emp) ? result.fro_norm_emp : nothing
        lines!(norm_ax, lag_times, primary_norm; color=METHOD_COLOR, linewidth=2.5)
        secondary_norm === nothing || lines!(norm_ax, lag_times, secondary_norm; color=:darkorange2, linewidth=2.0, linestyle=:dash)
        hlines!(norm_ax, [0.0]; color=:gray60, linestyle=:dash)
        grid_row += output_dim
    end

    Label(fig[0, :], "Brusselator Observed Residuals", fontsize=30)
    return fig
end

function resample_to_axis(source_axis::AbstractVector{<:Real},
                          source_values::AbstractVector{<:Real},
                          target_axis::AbstractVector{<:Real})
    length(source_axis) == length(source_values) || error("Source axis and values must have the same length.")
    length(source_axis) > 0 || return Float64[]
    result = Vector{Float64}(undef, length(target_axis))
    source_axis_f = Float64.(source_axis)
    source_values_f = Float64.(source_values)
    first_x = source_axis_f[1]
    last_x = source_axis_f[end]

    @inbounds for i in eachindex(target_axis)
        x = Float64(target_axis[i])
        if x <= first_x
            result[i] = source_values_f[1]
        elseif x >= last_x
            result[i] = source_values_f[end]
        else
            upper = searchsortedfirst(source_axis_f, x)
            lower = upper - 1
            x0 = source_axis_f[lower]
            x1 = source_axis_f[upper]
            y0 = source_values_f[lower]
            y1 = source_values_f[upper]
            w = (x - x0) / (x1 - x0)
            result[i] = (1 - w) * y0 + w * y1
        end
    end
    return result
end

function single_method_figure(method_diag,
                              method_traj::AbstractMatrix,
                              method_meta::Dict{String,Any},
                              method_metrics,
                              phi_sigma_info,
                              true_diag,
                              true_traj::AbstractMatrix,
                              true_time_axis::AbstractVector{<:Real})
    fig = Figure(size=(2000, 2400))
    ax_pdf_x = Axis(fig[1, 1], xlabel="x", ylabel="density", title="Marginal PDF of x")
    ax_pdf_y = Axis(fig[1, 2], xlabel="y", ylabel="density", title="Marginal PDF of y")
    ax_acf_x = Axis(fig[1, 3], xlabel="lag", ylabel="ACF", title="ACF of x")
    ax_acf_y = Axis(fig[1, 4], xlabel="lag", ylabel="ACF", title="ACF of y")

    ax_joint_true = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Observed joint PDF")
    ax_joint_model = Axis(fig[2, 2], xlabel="x", ylabel="y", title="Reduced-model joint PDF")
    ax_phase = Axis(fig[2, 3], xlabel="x", ylabel="y", title="Phase portrait")
    ax_ccf = Axis(fig[2, 4], xlabel="lag", ylabel="CCF", title="Cross-correlation")

    ax_time_x = Axis(fig[3, 1:2], xlabel="t", ylabel="x", title="Representative trajectory: x")
    ax_time_y = Axis(fig[3, 3:4], xlabel="t", ylabel="y", title="Representative trajectory: y")

    ax_summary = Axis(fig[4, 1], title="Run summary")
    hidedecorations!(ax_summary)
    hidespines!(ax_summary)
    ax_stein = Axis(fig[4, 2], xlabel="j", ylabel="i", title="Stein matrix V", aspect=DataAspect())
    ax_phi_s = Axis(fig[4, 3], xlabel="j", ylabel="i", title="Φ_S", aspect=DataAspect())
    ax_phi_a = Axis(fig[4, 4], xlabel="j", ylabel="i", title="Φ_A", aspect=DataAspect())

    lines!(ax_pdf_x, true_diag.pdf_x.x, true_diag.pdf_x.density, color=OBS_COLOR, linewidth=3, label="obs")
    lines!(ax_pdf_x, method_diag.pdf_x.x, method_diag.pdf_x.density, color=METHOD_COLOR, linewidth=2.5, label="model")
    lines!(ax_pdf_y, true_diag.pdf_y.x, true_diag.pdf_y.density, color=OBS_COLOR, linewidth=3, label="obs")
    lines!(ax_pdf_y, method_diag.pdf_y.x, method_diag.pdf_y.density, color=METHOD_COLOR, linewidth=2.5, label="model")

    lines!(ax_acf_x, true_diag.lag_axis, true_diag.acf_x, color=OBS_COLOR, linewidth=3, label="obs")
    lines!(ax_acf_x, true_diag.lag_axis, resample_to_axis(method_diag.lag_axis, method_diag.acf_x, true_diag.lag_axis), color=METHOD_COLOR, linewidth=2.5, label="model")
    lines!(ax_acf_y, true_diag.lag_axis, true_diag.acf_y, color=OBS_COLOR, linewidth=3, label="obs")
    lines!(ax_acf_y, true_diag.lag_axis, resample_to_axis(method_diag.lag_axis, method_diag.acf_y, true_diag.lag_axis), color=METHOD_COLOR, linewidth=2.5, label="model")
    lines!(ax_ccf, true_diag.ccf_axis, true_diag.ccf, color=OBS_COLOR, linewidth=3, label="obs")
    lines!(ax_ccf, true_diag.ccf_axis, resample_to_axis(method_diag.ccf_axis, method_diag.ccf, true_diag.ccf_axis), color=METHOD_COLOR, linewidth=2.5, label="model")

    heatmap!(ax_joint_true, true_diag.pdf_xy.x, true_diag.pdf_xy.y, true_diag.pdf_xy.density, colormap=:viridis)
    heatmap!(ax_joint_model, method_diag.pdf_xy.x, method_diag.pdf_xy.y, method_diag.pdf_xy.density, colormap=:viridis)
    lines!(ax_phase, true_traj[1, 1:true_diag.phase_count], true_traj[2, 1:true_diag.phase_count], color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_phase, method_traj[1, 1:method_diag.phase_count], method_traj[2, 1:method_diag.phase_count], color=METHOD_COLOR, linewidth=2.0, label="model")

    obs_time = Float64.(true_time_axis)
    model_time = collect(range(first(obs_time), last(obs_time), length=size(method_traj, 2)))
    model_x = resample_to_axis(model_time, method_traj[1, :], obs_time)
    model_y = resample_to_axis(model_time, method_traj[2, :], obs_time)
    lines!(ax_time_x, obs_time, true_traj[1, :], color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_time_x, obs_time, model_x, color=METHOD_COLOR, linewidth=2.0, label="model")
    lines!(ax_time_y, obs_time, true_traj[2, :], color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_time_y, obs_time, model_y, color=METHOD_COLOR, linewidth=2.0, label="model")

    draw_matrix_panel!(ax_stein, phi_sigma_info.V, "Stein matrix V")
    draw_matrix_panel!(ax_phi_s, phi_sigma_info.Phi_symmetric, "Φ_S")
    draw_matrix_panel!(ax_phi_a, phi_sigma_info.Phi_antisymmetric, "Φ_A")

    summary_lines = [
        "training_time_seconds = $(round(method_meta["training_time_seconds"], digits=2))",
        "noise_scale = $(method_meta["noise_scale"])",
        "clusters = $(phi_sigma_info.n_states)",
        "regularization_shift = $(round(phi_sigma_info.regularization_shift, digits=8))",
        "KLxy = $(round(method_metrics.kl_xy, digits=6))",
        "RMSE ACF x = $(round(method_metrics.rmse_acf_x, digits=6))",
        "RMSE ACF y = $(round(method_metrics.rmse_acf_y, digits=6))",
        "RMSE CCF = $(round(method_metrics.rmse_ccf, digits=6))",
    ]
    text!(ax_summary, 0.0, 1.0; text=join(summary_lines, "\n"), align=(:left, :top), space=:relative, fontsize=22)

    for axis in (ax_pdf_x, ax_pdf_y, ax_acf_x, ax_acf_y, ax_ccf, ax_phase, ax_time_x, ax_time_y)
        axislegend(axis, position=:rt)
    end

    Label(fig[0, :], "Brusselator Reduced Langevin Validation", fontsize=28)
    return fig
end

struct JointPairScoreSliceModel{M,T}
    model::M
    noise_scale::T
    tau::T
end

Flux.@functor JointPairScoreSliceModel

function JointPairScoreSliceModel(model, noise_scale::Real, tau::Real)
    return JointPairScoreSliceModel{typeof(model),Float32}(model, Float32(noise_scale), Float32(tau))
end

function Flux.testmode!(wrapper::JointPairScoreSliceModel, mode=true)
    Flux.testmode!(wrapper.model, mode)
    return wrapper
end

function ScoreEstimation._move_to_gpu(wrapper::JointPairScoreSliceModel, cuda)
    return JointPairScoreSliceModel(
        ScoreEstimation._move_to_gpu(wrapper.model, cuda),
        wrapper.noise_scale,
        wrapper.tau,
    )
end

function (wrapper::JointPairScoreSliceModel)(y::AbstractMatrix)
    tau_row = similar(y, eltype(y), 1, size(y, 2))
    fill!(tau_row, convert(eltype(y), wrapper.tau))
    input = vcat(y, tau_row)
    pred = wrapper.model(input)
    scale = -inv(convert(eltype(y), wrapper.noise_scale))
    return scale .* pred
end

const JOINT_PAIR_PROJECTIONS = (
    (1, 3, "x0_1 vs xt_1"),
    (2, 4, "x0_2 vs xt_2"),
    (1, 4, "x0_1 vs xt_2"),
    (2, 3, "x0_2 vs xt_1"),
)

_joint_swish(x) = x .* Flux.sigmoid.(x)

struct JointResidualScoreModel{M,S,V,T,F}
    correction_model::M
    stationary_score_model::S
    mean::V
    scale::V
    noise_scale::T
    time_frequencies::F
end

Flux.@functor JointResidualScoreModel

function Flux.testmode!(model::JointResidualScoreModel, mode=true)
    Flux.testmode!(model.correction_model, mode)
    model.stationary_score_model === nothing || Flux.testmode!(model.stationary_score_model, mode)
    return model
end

function ScoreEstimation._move_to_gpu(model::JointResidualScoreModel, cuda)
    return JointResidualScoreModel(
        ScoreEstimation._move_to_gpu(model.correction_model, cuda),
        model.stationary_score_model === nothing ? nothing : ScoreEstimation._move_to_gpu(model.stationary_score_model, cuda),
        Base.invokelatest(cuda.cu, model.mean),
        Base.invokelatest(cuda.cu, model.scale),
        model.noise_scale,
        model.time_frequencies,
    )
end

function joint_feature_map(y::AbstractMatrix,
                           tau::AbstractMatrix,
                           time_frequencies::AbstractVector)
    midpoint = @views 0.5f0 .* (y[1:2, :] .+ y[3:4, :])
    delta = @views y[3:4, :] .- y[1:2, :]
    T = eltype(y)
    harmonic_blocks = [
        begin
        angle = @. (2 * T(pi) * convert(T, freq)) * tau
            vcat(sin.(angle), cos.(angle))
        end
        for freq in time_frequencies
    ]
    return isempty(harmonic_blocks) ? vcat(y, midpoint, delta, tau) : vcat(y, midpoint, delta, tau, harmonic_blocks...)
end

function build_joint_correction_mlp(hidden::Vector{Int}, time_frequencies::AbstractVector)
    input_dim = 9 + 2 * length(time_frequencies)
    sizes = vcat(input_dim, hidden, 4)
    layers = Vector{Any}(undef, length(sizes) - 1)
    for idx in 1:(length(sizes) - 2)
        layers[idx] = Flux.Dense(sizes[idx], sizes[idx + 1], _joint_swish)
    end
    layers[end] = Flux.Dense(sizes[end - 1], sizes[end])
    return Flux.Chain(layers...)
end

function JointResidualScoreModel(stationary_score_model,
                                 mean::AbstractVector,
                                 scale::AbstractVector,
                                 noise_scale::Real,
                                 hidden::Vector{Int};
                                 time_frequencies::AbstractVector=[1.0f0, 2.0f0, 4.0f0, 8.0f0])
    correction_model = build_joint_correction_mlp(hidden, time_frequencies)
    return JointResidualScoreModel(
        correction_model,
        stationary_score_model,
        Float32.(collect(mean)),
        Float32.(collect(scale)),
        Float32(noise_scale),
        Float32.(collect(time_frequencies)),
    )
end

function (model::JointResidualScoreModel)(input::AbstractMatrix)
    y = @view input[1:4, :]
    tau = @view input[5:5, :]
    features = joint_feature_map(y, tau, model.time_frequencies)
    return model.correction_model(features)
end

function pair_normalization_stats(samples::AbstractMatrix)
    μ2 = vec(mean(samples; dims=2))
    σ2 = vec(std(samples; dims=2))
    σ2 .= max.(σ2, 1.0f-6)
    μ4 = Float32.(vcat(μ2, μ2))
    σ4 = Float32.(vcat(σ2, σ2))
    return μ4, σ4
end

function normalize_pair_samples(samples::AbstractMatrix, μ4::AbstractVector, σ4::AbstractVector)
    return (Float32.(samples) .- reshape(Float32.(μ4), :, 1)) ./ reshape(Float32.(σ4), :, 1)
end

function denormalize_pair_samples(samples::AbstractMatrix, μ4::AbstractVector, σ4::AbstractVector)
    return Float32.(samples) .* reshape(Float32.(σ4), :, 1) .+ reshape(Float32.(μ4), :, 1)
end

function pair_samples_for_lag(correlated::AbstractMatrix, lag_step::Integer)
    n_time = size(correlated, 2)
    lag = Int(lag_step)
    lag < n_time || error("Lag $(lag) is too large for trajectory length $(n_time).")
    n_pairs = n_time - lag
    pairs = Matrix{Float32}(undef, 4, n_pairs)
    @views pairs[1:2, :] .= correlated[:, 1:n_pairs]
    @views pairs[3:4, :] .= correlated[:, (1 + lag):(lag + n_pairs)]
    return pairs
end

function representative_lag_steps(max_lag_step::Integer, fractions)
    candidates = Int[]
    for frac in fractions
        push!(candidates, clamp(round(Int, Float64(frac) * max_lag_step), 1, max_lag_step))
    end
    return unique(sort(candidates))
end

function build_joint_training_batch(correlated::AbstractMatrix,
                                    lag_steps::AbstractVector{<:Integer},
                                    lag_cdf::Union{Nothing,AbstractVector},
                                    μ4::AbstractVector,
                                    σ4::AbstractVector,
                                    batch_size::Integer,
                                    rng::AbstractRNG)
    n_time = size(correlated, 2)
    max_lag_step = Int(last(lag_steps))
    y = Matrix{Float32}(undef, 4, batch_size)
    tau = Matrix{Float32}(undef, 1, batch_size)

    @inbounds for j in 1:batch_size
        lag = if lag_cdf === nothing
            rand(rng, lag_steps)
        else
            lag_steps[searchsortedfirst(lag_cdf, rand(rng))]
        end
        start_idx = rand(rng, 1:(n_time - lag))
        x0 = @view correlated[:, start_idx]
        xt = @view correlated[:, start_idx + lag]
        y[1, j] = (Float32(x0[1]) - μ4[1]) / σ4[1]
        y[2, j] = (Float32(x0[2]) - μ4[2]) / σ4[2]
        y[3, j] = (Float32(xt[1]) - μ4[3]) / σ4[3]
        y[4, j] = (Float32(xt[2]) - μ4[4]) / σ4[4]
        tau[1, j] = Float32(lag / max_lag_step)
    end

    return y, tau
end

function joint_dsm_loss(nn,
                        y_clean::AbstractMatrix,
                        tau::AbstractMatrix,
                        noise_scale::Real;
                        use_gpu::Bool=false,
                        mean_regularization::Real=0.0,
                        stein_regularization::Real=0.0)
    ϵ = randn(Float32, size(y_clean))
    y_noisy = @. Float32(y_clean) + Float32(noise_scale) * ϵ
    input = vcat(y_noisy, Float32.(tau))
    input_dev = ScoreEstimation._to_device(input; use_gpu=use_gpu)
    ϵ_dev = ScoreEstimation._to_device(ϵ; use_gpu=use_gpu)
    y_noisy_dev = view(input_dev, 1:4, :)

    pred, loss = Flux.withgradient(nn) do model
        pred_local = model(input_dev)
        local_loss = Flux.mse(pred_local, ϵ_dev) + ScoreEstimation._score_moment_penalty(
            pred_local,
            y_noisy_dev,
            noise_scale;
            mean_weight=mean_regularization,
            stein_weight=stein_regularization,
        )
        return pred_local, local_loss
    end
    return pred, loss
end

function estimate_joint_dsm_loss(nn,
                                 correlated::AbstractMatrix,
                                 lag_steps::AbstractVector{<:Integer},
                                 μ4::AbstractVector,
                                 σ4::AbstractVector,
                                 noise_scale::Real;
                                 batch_size::Integer,
                                 eval_batches_per_lag::Integer,
                                 use_gpu::Bool=false,
                                 seed::Integer=0)
    rng = MersenneTwister(seed)
    losses = Vector{Float64}(undef, length(lag_steps))
    for (idx, lag_step) in enumerate(lag_steps)
        lag_losses = zeros(Float64, eval_batches_per_lag)
        for batch_idx in 1:eval_batches_per_lag
            y_clean, tau = build_joint_training_batch(correlated, [lag_step], nothing, μ4, σ4, batch_size, rng)
            ϵ = randn(rng, Float32, size(y_clean))
            y_noisy = @. y_clean + Float32(noise_scale) * ϵ
            input = vcat(y_noisy, tau)
            input_dev = ScoreEstimation._to_device(input; use_gpu=use_gpu)
            ϵ_dev = ScoreEstimation._to_device(ϵ; use_gpu=use_gpu)
            pred = nn(input_dev)
            lag_losses[batch_idx] = Float64(Flux.mse(pred, ϵ_dev))
        end
        losses[idx] = mean(lag_losses)
    end
    return losses
end

function score_matrix(wrapper::JointPairScoreSliceModel,
                      samples::AbstractMatrix;
                      batch_size::Integer=8192,
                      use_gpu::Bool=false)
    n_samples = size(samples, 2)
    scores = Matrix{Float32}(undef, 4, n_samples)
    model_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(wrapper), CUDA)
    else
        deepcopy(wrapper)
    end
    Flux.testmode!(model_eval)

    start_idx = 1
    while start_idx <= n_samples
        stop_idx = min(start_idx + batch_size - 1, n_samples)
        batch = @view samples[:, start_idx:stop_idx]
        batch_dev = ScoreEstimation._to_device(Float32.(batch); use_gpu=use_gpu)
        @views scores[:, start_idx:stop_idx] .= Array(model_eval(batch_dev))
        start_idx = stop_idx + 1
    end
    return scores
end

function stein_matrix(samples::AbstractMatrix,
                      wrapper::JointPairScoreSliceModel;
                      batch_size::Integer=8192,
                      use_gpu::Bool=false)
    score = score_matrix(wrapper, samples; batch_size=batch_size, use_gpu=use_gpu)
    return (Float64.(score) * transpose(Float64.(samples))) ./ size(samples, 2)
end

function contour_levels_from_pdf(pdf::BivariatePDFEstimate, nlevels::Integer)
    peak = maximum(pdf.density)
    peak > 0 || return Float64[]
    lo = 0.15 * peak
    hi = 0.9 * peak
    return collect(range(lo, hi; length=max(nlevels, 2)))
end

function _pdf_mass(pdf::BivariatePDFEstimate)
    nx = length(pdf.x)
    ny = length(pdf.y)
    (nx <= 1 || ny <= 1) && return copy(pdf.density)
    dx = pdf.x[2] - pdf.x[1]
    dy = pdf.y[2] - pdf.y[1]
    mass = max.(pdf.density, 0.0) .* (dx * dy)
    total = sum(mass)
    total > 0 || return fill(1.0 / length(mass), size(mass))
    return mass ./ total
end

function _discrete_kl(p::AbstractArray, q::AbstractArray; floor::Float64=1.0e-12)
    p_safe = max.(p, floor)
    q_safe = max.(q, floor)
    return sum(p_safe .* log.(p_safe ./ q_safe))
end

function _l1_distance(p::AbstractArray, q::AbstractArray)
    return sum(abs.(p .- q))
end

function pair_pdf_metrics(obs_pdf::BivariatePDFEstimate, model_pdf::BivariatePDFEstimate)
    obs_mass = _pdf_mass(obs_pdf)
    model_mass = _pdf_mass(model_pdf)
    return (
        kl=_discrete_kl(obs_mass, model_mass),
        l1=_l1_distance(obs_mass, model_mass),
    )
end

function vector_value_range(samples::AbstractVector{<:Real})
    return ScoreEstimation.determine_value_range(reshape(collect(samples), 1, :))
end

function joint_score_figure(rows,
                            training_losses::AbstractVector,
                            stage_offsets::AbstractVector,
                            stage_noise_scales::AbstractVector,
                            lag_steps::AbstractVector,
                            lag_times::AbstractVector,
                            loss_by_lag::AbstractVector,
                            tD::Real)
    nrows = length(rows) + 1
    fig = Figure(size=(2600, 700 * nrows))

    for (row_idx, row) in enumerate(rows)
        for (col_idx, projection) in enumerate(row.projections)
            ax = Axis(fig[row_idx, col_idx], xlabel=projection.xlabel, ylabel=projection.ylabel,
                title="$(projection.label), t=$(round(row.lag_time, digits=2))")
            heatmap!(ax, projection.obs_pdf.x, projection.obs_pdf.y, projection.obs_pdf.density; colormap=:magma)
            levels = contour_levels_from_pdf(projection.model_pdf, row.contour_levels)
            isempty(levels) || contour!(ax, projection.model_pdf.x, projection.model_pdf.y, projection.model_pdf.density;
                levels=levels, color=:cyan, linewidth=2)
            text!(ax, 0.03, 0.97;
                text="KL=$(round(projection.metrics.kl, digits=4))\nL1=$(round(projection.metrics.l1, digits=4))",
                space=:relative, align=(:left, :top), color=:white, fontsize=18)
        end

        ax_stein = Axis(fig[row_idx, 5], xlabel="j", ylabel="i",
            title="Stein V, t=$(round(row.lag_time, digits=2))", aspect=DataAspect())
        draw_matrix_panel!(ax_stein, row.stein_matrix, "Stein V, ||V+I||=$(round(row.stein_error, digits=4))")
    end

    loss_ax = Axis(fig[nrows, 1:2], xlabel="epoch", ylabel="DSM loss", title="Joint-score training loss")
    lines!(loss_ax, 1:length(training_losses), training_losses, color=:dodgerblue3, linewidth=2.5)
    for (idx, offset) in enumerate(stage_offsets)
        vlines!(loss_ax, [offset + 1], color=:gray60, linestyle=:dash)
        text!(loss_ax, offset + 1, maximum(training_losses); text="σ=$(stage_noise_scales[idx])", align=(:left, :top), fontsize=16)
    end

    lag_ax = Axis(fig[nrows, 3:4], xlabel="lag time", ylabel="DSM loss", title="Loss by lag")
    lines!(lag_ax, lag_times, loss_by_lag, color=:darkorange2, linewidth=2.5)
    vlines!(lag_ax, [tD], color=:gray50, linestyle=:dash)

    summary_ax = Axis(fig[nrows, 5], title="Summary")
    hidedecorations!(summary_ax)
    hidespines!(summary_ax)
    summary_lines = String[
        "tD = $(round(tD, digits=4))",
        "max lag loss = $(round(maximum(loss_by_lag), digits=5))",
        "mean lag loss = $(round(mean(loss_by_lag), digits=5))",
    ]
    for row in rows
        mean_kl = mean([projection.metrics.kl for projection in row.projections])
        push!(summary_lines, "t=$(round(row.lag_time, digits=2)): mean KL=$(round(mean_kl, digits=4))")
        push!(summary_lines, "t=$(round(row.lag_time, digits=2)): ||V+I||=$(round(row.stein_error, digits=4))")
    end
    text!(summary_ax, 0.0, 1.0; text=join(summary_lines, "\n"), space=:relative, align=(:left, :top), fontsize=18)

    Label(fig[0, :], "Brusselator Joint-Score Diagnostics", fontsize=28)
    return fig
end

evaluate_observable_device(obs::ObservableSpec, samples) = obs.eval_batch(samples)

function empirical_observable_correlation_tensor(obs::ObservableSpec,
                                                 trajectory::AbstractMatrix,
                                                 lag_steps::AbstractVector{<:Integer})
    n_time = size(trajectory, 2)
    dim = size(trajectory, 1)
    output_dim = length(obs.output_labels)
    values = Array{Float64}(undef, output_dim, dim, length(lag_steps))

    for (idx, lag_step) in enumerate(lag_steps)
        lag = Int(lag_step)
        n_pairs = n_time - lag
        n_pairs > 0 || error("Lag $(lag) leaves no samples for observable correlation.")
        x0 = @view trajectory[:, 1:n_pairs]
        xt = @view trajectory[:, (1 + lag):(lag + n_pairs)]
        phi_xt = Float64.(evaluate_observable(obs, xt))
        @views values[:, :, idx] .= (phi_xt * transpose(Float64.(x0))) ./ n_pairs
    end

    return values
end

function empirical_observable_correlation_tensor(obs::ObservableSpec,
                                                 trajectories::Array{Float32,3},
                                                 lag_steps::AbstractVector{<:Integer})
    n_time = size(trajectories, 2)
    n_ensembles = size(trajectories, 3)
    dim = size(trajectories, 1)
    output_dim = length(obs.output_labels)
    values = Array{Float64}(undef, output_dim, dim, length(lag_steps))

    for (idx, lag_step) in enumerate(lag_steps)
        lag = Int(lag_step)
        n_pairs = n_time - lag
        n_pairs > 0 || error("Lag $(lag) leaves no samples for observable correlation.")
        accum = zeros(Float64, output_dim, dim)
        for ensemble_idx in 1:n_ensembles
            x0 = @view trajectories[:, 1:n_pairs, ensemble_idx]
            xt = @view trajectories[:, (1 + lag):(lag + n_pairs), ensemble_idx]
            phi_xt = Float64.(evaluate_observable(obs, xt))
            accum .+= (phi_xt * transpose(Float64.(x0))) ./ n_pairs
        end
        @views values[:, :, idx] .= accum ./ n_ensembles
    end

    return values
end

function coordinate_affine_closure_metrics(correlated::AbstractMatrix,
                                           lag_steps::AbstractVector{<:Integer})
    μ = vec(mean(Float64.(correlated); dims=2))
    centered = Float64.(correlated) .- reshape(μ, :, 1)
    C00 = (centered * transpose(centered)) ./ size(centered, 2)
    C00_inv = pinv(C00)
    rel_rmse = Vector{Float64}(undef, length(lag_steps))
    A_mats = Array{Float64}(undef, 2, 2, length(lag_steps))
    for (idx, lag_step) in enumerate(lag_steps)
        lag = Int(lag_step)
        n_pairs = size(correlated, 2) - lag
        n_pairs > 0 || error("Lag $(lag) leaves no samples for affine-closure diagnostics.")
        x0 = Float64.(correlated[:, 1:n_pairs])
        xt = Float64.(correlated[:, (1 + lag):(lag + n_pairs)])
        x0c = x0 .- reshape(μ, :, 1)
        xtc = xt .- reshape(μ, :, 1)
        Ct = (xtc * transpose(x0c)) ./ n_pairs
        A = Ct * C00_inv
        pred = reshape(μ, :, 1) .+ A * x0c
        rel_rmse[idx] = sqrt(mean(abs2, xt .- pred) / max(mean(abs2, xt .- reshape(μ, :, 1)), eps(Float64)))
        @views A_mats[:, :, idx] .= A
    end
    return (rel_rmse=rel_rmse, A=A_mats)
end

function build_lag_steps(data, tD_override::Real, lag_step_stride::Integer; include_zero::Bool=false)
    lag_max_step = resolve_lag_max_step(data, tD_override)
    lag_start = include_zero ? 0 : max(1, lag_step_stride)
    lag_steps = collect(lag_start:max(Int(lag_step_stride), 1):lag_max_step)
    isempty(lag_steps) && push!(lag_steps, lag_max_step)
    last(lag_steps) == lag_max_step || push!(lag_steps, lag_max_step)
    return unique(sort(Int.(lag_steps)))
end

function resolve_delta_m_rollout_lag_steps(data,
                                           training_lag_steps::AbstractVector{<:Integer},
                                           rollout_cfg::Dict{String,Any},
                                           lag_step_stride::Integer)
    eval_tD_override = get(rollout_cfg, "evaluation_tD_override", nothing)
    if isnothing(eval_tD_override)
        return vcat(0, Int.(training_lag_steps))
    end
    return build_lag_steps(data, Float64(eval_tD_override), lag_step_stride; include_zero=true)
end

function read_observed_residual_targets(path::AbstractString,
                                        observables::AbstractVector{ObservableSpec},
                                        lag_steps::AbstractVector{<:Integer})
    targets = Dict{String,Any}()
    h5open(path, "r") do file
        available_steps = Int.(read(file["lag_steps"]))
        available_times = Float64.(read(file["lag_times"]))
        lookup = Dict(step => idx for (idx, step) in enumerate(available_steps))
        target_times = Float64.(lag_steps) .* (available_times[2] - available_times[1])
        for obs in observables
            group = file[slugify(obs.name)]
            Ephi = Float64.(read(group["Ephi"]))
            indices = [lookup[Int(step)] for step in lag_steps]
            targets[obs.name] = (
                E_obs=Ephi[:, :, indices],
                output_labels=obs.output_labels,
            )
        end
        targets["_lag_times"] = target_times
    end
    return targets
end

struct ConditionalScoreWrapper{J,S,V,T}
    joint_model::J
    stationary_score::S
    pair_mean::V
    pair_scale::V
    noise_scale::T
    tD::T
end

Flux.@functor ConditionalScoreWrapper

function Flux.testmode!(wrapper::ConditionalScoreWrapper, mode=true)
    Flux.testmode!(wrapper.joint_model, mode)
    Flux.testmode!(wrapper.stationary_score, mode)
    return wrapper
end

function ScoreEstimation._move_to_gpu(wrapper::ConditionalScoreWrapper, cuda)
    return ConditionalScoreWrapper(
        ScoreEstimation._move_to_gpu(wrapper.joint_model, cuda),
        ScoreEstimation._move_to_gpu(wrapper.stationary_score, cuda),
        Base.invokelatest(cuda.cu, wrapper.pair_mean),
        Base.invokelatest(cuda.cu, wrapper.pair_scale),
        wrapper.noise_scale,
        wrapper.tD,
    )
end

function build_conditional_score_wrapper(paths)
    stationary_score, _ = build_score_wrapper(paths.score_model_bson)
    joint_model, joint_meta = load_model(paths.joint_score_model_bson)
    Flux.testmode!(joint_model)
    return ConditionalScoreWrapper(
        joint_model,
        stationary_score,
        Float32.(joint_meta["mean"]),
        Float32.(joint_meta["scale"]),
        Float32(joint_meta["noise_scale"]),
        Float32(joint_meta["tD"]),
    )
end

function conditional_score(wrapper::ConditionalScoreWrapper,
                           x0,
                           xt,
                           lag_time::Real)
    T = eltype(x0)
    μ = reshape(wrapper.pair_mean, :, 1)
    σ = reshape(wrapper.pair_scale, :, 1)
    y0 = (x0 .- @view(μ[1:2, :])) ./ @view(σ[1:2, :])
    yt = (xt .- @view(μ[3:4, :])) ./ @view(σ[3:4, :])
    tau_norm = similar(x0, T, 1, size(x0, 2))
    fill!(tau_norm, convert(T, Float32(lag_time) / wrapper.tD))
    input = vcat(y0, yt, tau_norm)
    eps_hat = wrapper.joint_model(input)
    joint_score = (-eps_hat ./ convert(T, wrapper.noise_scale))
    joint_x0 = @view(joint_score[1:2, :]) ./ @view(σ[1:2, :])
    stationary = wrapper.stationary_score(x0)
    return joint_x0 .- stationary
end

@inline _inverse_softplus(x::Real) = x > 20 ? Float32(x) : Float32(log(expm1(max(x, 1.0f-6))))

struct DeltaMMobilityField{M,V,P,T}
    backbone::M
    state_mean::V
    state_scale::V
    feature_mean::V
    feature_scale::V
    quad_center::V
    phi_baseline::P
    theta_baseline::V
    residual_scale::V
    eps_floor::T
end

Flux.@functor DeltaMMobilityField

function Flux.testmode!(model::DeltaMMobilityField, mode=true)
    Flux.testmode!(model.backbone, mode)
    return model
end

function ScoreEstimation._move_to_gpu(model::DeltaMMobilityField, cuda)
    return DeltaMMobilityField(
        ScoreEstimation._move_to_gpu(model.backbone, cuda),
        Base.invokelatest(cuda.cu, model.state_mean),
        Base.invokelatest(cuda.cu, model.state_scale),
        Base.invokelatest(cuda.cu, model.feature_mean),
        Base.invokelatest(cuda.cu, model.feature_scale),
        Base.invokelatest(cuda.cu, model.quad_center),
        model.phi_baseline,
        Base.invokelatest(cuda.cu, model.theta_baseline),
        Base.invokelatest(cuda.cu, model.residual_scale),
        model.eps_floor,
    )
end

function DeltaMMobilityField(state_mean::AbstractVector,
                             state_scale::AbstractVector,
                             feature_mean::AbstractVector,
                             feature_scale::AbstractVector,
                             quad_center::AbstractVector,
                             hidden::AbstractVector{<:Integer},
                             phi_baseline::AbstractMatrix,
                             sigma_baseline::AbstractMatrix,
                             omega_baseline::Real,
                             residual_scale::AbstractVector;
                             eps_floor::Real=1.0f-4)
    sizes = vcat(7, Int.(hidden), 4)
    layers = Vector{Any}(undef, length(sizes) - 1)
    for idx in 1:(length(sizes) - 2)
        layers[idx] = Flux.Dense(sizes[idx], sizes[idx + 1], _joint_swish)
    end
    layers[end] = Flux.Dense(sizes[end - 1], sizes[end])
    chain = Flux.Chain(layers...)
    output = chain.layers[end]
    output.weight .= 0.0f0
    output.bias .= 0.0f0
    return DeltaMMobilityField(
        Flux.f32(chain),
        Float32.(collect(state_mean)),
        Float32.(collect(state_scale)),
        Float32.(collect(feature_mean)),
        Float32.(collect(feature_scale)),
        Float32.(collect(quad_center)),
        Float32.(phi_baseline),
        Float32[
            _inverse_softplus(max(Float32(sigma_baseline[1, 1]) - Float32(eps_floor), 1.0f-5)),
            Float32(sigma_baseline[2, 1]),
            _inverse_softplus(max(Float32(sigma_baseline[2, 2]) - Float32(eps_floor), 1.0f-5)),
            Float32(omega_baseline),
        ],
        Float32.(collect(residual_scale)),
        Float32(eps_floor),
    )
end

function mobility_feature_matrix(model::DeltaMMobilityField,
                                 x,
                                 stationary_score)
    T = eltype(x)
    μ = reshape(model.state_mean, :, 1)
    σ = reshape(model.state_scale, :, 1)
    xc = x .- reshape(model.quad_center, :, 1)
    xnorm = (x .- μ) ./ σ
    q1 = reshape(@view(xc[1, :]) .^ 2, 1, :)
    q2 = reshape(@view(xc[1, :]) .* @view(xc[2, :]), 1, :)
    q3 = reshape(@view(xc[2, :]) .^ 2, 1, :)
    features = vcat(xnorm, stationary_score, q1, q2, q3)
    fμ = reshape(model.feature_mean, :, 1)
    fσ = reshape(model.feature_scale, :, 1)
    return (features .- fμ) ./ fσ
end

function mobility_components(model::DeltaMMobilityField,
                             x,
                             stationary_score)
    raw = model.backbone(mobility_feature_matrix(model, x, stationary_score))
    T = eltype(raw)
    eps_floor = convert(T, model.eps_floor)
    θ0 = reshape(model.theta_baseline, :, 1)
    amp = reshape(model.residual_scale, :, 1)
    residual = amp .* tanh.(raw)
    θ11 = @view(θ0[1, :]) .+ @view(residual[1, :])
    l21 = @view(θ0[2, :]) .+ @view(residual[2, :])
    θ22 = @view(θ0[3, :]) .+ @view(residual[3, :])
    ω = @view(θ0[4, :]) .+ @view(residual[4, :])
    l11 = Flux.softplus.(θ11) .+ eps_floor
    l22 = Flux.softplus.(θ22) .+ eps_floor
    m11 = l11 .* l11
    sym12 = l11 .* l21
    m12 = sym12 .+ ω
    m21 = sym12 .- ω
    m22 = l21 .* l21 .+ l22 .* l22
    phi = model.phi_baseline
    delta11 = m11 .- convert(T, phi[1, 1])
    delta12 = m12 .- convert(T, phi[1, 2])
    delta21 = m21 .- convert(T, phi[2, 1])
    delta22 = m22 .- convert(T, phi[2, 2])
    return (
        sigma11=l11,
        sigma21=l21,
        sigma22=l22,
        omega=ω,
        m11=m11,
        m12=m12,
        m21=m21,
        m22=m22,
        residual11=@view(residual[1, :]),
        residual21=@view(residual[2, :]),
        residual22=@view(residual[3, :]),
        residualω=@view(residual[4, :]),
        delta11=delta11,
        delta12=delta12,
        delta21=delta21,
        delta22=delta22,
    )
end

function effective_delta_score(components, scond)
    s1 = @view scond[1, :]
    s2 = @view scond[2, :]
    eff1 = components.delta11 .* s1 .+ components.delta21 .* s2
    eff2 = components.delta12 .* s1 .+ components.delta22 .* s2
    return vcat(reshape(eff1, 1, :), reshape(eff2, 1, :))
end

function mobility_state_at_point(model::DeltaMMobilityField,
                                 score_wrapper,
                                 x::AbstractVector)
    xmat = reshape(x, :, 1)
    score = score_wrapper(xmat)
    comps = mobility_components(model, xmat, score)
    T = eltype(xmat)
    M = T[
        comps.m11[1] comps.m12[1];
        comps.m21[1] comps.m22[1];
    ]
    Sigma = T[
        comps.sigma11[1] zero(T);
        comps.sigma21[1] comps.sigma22[1];
    ]
    Delta = T[
        comps.delta11[1] comps.delta12[1];
        comps.delta21[1] comps.delta22[1];
    ]
    sym12 = 0.5 .* (M[1, 2] + M[2, 1])
    lambda_min = 0.5 .* ((M[1, 1] + M[2, 2]) - sqrt(max((M[1, 1] - M[2, 2])^2 + 4 .* (sym12^2), zero(T))))
    return (
        M=M,
        Sigma=Sigma,
        Delta=Delta,
        score=vec(score),
        omega=comps.omega[1],
        lambda_min=lambda_min,
    )
end

function autodiff_mobility_divergence(model::DeltaMMobilityField,
                                      score_wrapper,
                                      x::AbstractVector{<:Real})
    f(z) = begin
        state = mobility_state_at_point(model, score_wrapper, z)
        T = eltype(z)
        return T[
            state.M[1, 1],
            state.M[1, 2],
            state.M[2, 1],
            state.M[2, 2],
        ]
    end
    jac = ForwardDiff.jacobian(f, Float64.(x))
    return Float64[
        jac[1, 1] + jac[2, 2],
        jac[3, 1] + jac[4, 2],
    ]
end

function point_mobility_diagnostics(model::DeltaMMobilityField,
                                    score_wrapper,
                                    x::AbstractVector{<:Real};
                                    divergence_mode::Symbol=:autodiff)
    state = mobility_state_at_point(model, score_wrapper, Float64.(x))
    divergence = if divergence_mode === :autodiff
        autodiff_mobility_divergence(model, score_wrapper, x)
    elseif divergence_mode === :none
        zeros(Float64, 2)
    else
        error("Unsupported pointwise divergence mode $(divergence_mode).")
    end
    score_term = Array(state.M * state.score)
    return (
        M=Float64.(state.M),
        D=Float64.(state.Sigma * transpose(state.Sigma)),
        Sigma=Float64.(state.Sigma),
        Delta=Float64.(state.Delta),
        score=Float64.(state.score),
        divergence=Float64.(divergence),
        score_term=Float64.(score_term),
        drift=Float64.(score_term .+ divergence),
        omega=Float64(state.omega),
        lambda_min=Float64(state.lambda_min),
    )
end

function sample_lagged_pairs(correlated::AbstractMatrix,
                             lag_step::Integer,
                             batch_size::Integer,
                             rng::AbstractRNG)
    n_time = size(correlated, 2)
    lag = Int(lag_step)
    n_time > lag || error("Lag $(lag) is too large for trajectory length $(n_time).")
    x0 = Matrix{Float32}(undef, 2, batch_size)
    xt = Matrix{Float32}(undef, 2, batch_size)
    @inbounds for j in 1:batch_size
        start_idx = rand(rng, 1:(n_time - lag))
        x0[1, j] = correlated[1, start_idx]
        x0[2, j] = correlated[2, start_idx]
        xt[1, j] = correlated[1, start_idx + lag]
        xt[2, j] = correlated[2, start_idx + lag]
    end
    return x0, xt
end

function lagged_pairs_matrix(correlated::AbstractMatrix,
                             lag_step::Integer;
                             max_pairs::Integer=0,
                             rng::Union{Nothing,AbstractRNG}=nothing)
    lag = Int(lag_step)
    n_pairs = size(correlated, 2) - lag
    n_pairs > 0 || error("Lag $(lag) leaves no pairs.")
    indices = if max_pairs > 0 && max_pairs < n_pairs
        rng === nothing && error("A random generator is required when subsampling lagged pairs.")
        sort(rand(rng, 1:n_pairs, max_pairs))
    else
        collect(1:n_pairs)
    end
    x0 = Float32.(correlated[:, indices])
    xt = Float32.(correlated[:, indices .+ lag])
    return x0, xt
end

function relative_frobenius_rmse(reference::AbstractArray, candidate::AbstractArray)
    numerator = mean([sum(abs2, reference[:, :, k] .- candidate[:, :, k]) for k in axes(reference, 3)])
    denominator = mean([sum(abs2, reference[:, :, k]) for k in axes(reference, 3)])
    return sqrt(numerator / max(denominator, eps(Float64)))
end

function relative_frobenius_rmse(reference::AbstractVector, candidate::AbstractVector)
    return sqrt(mean(abs2, reference .- candidate) / max(mean(abs2, reference), eps(Float64)))
end

function build_field_axes(samples::AbstractMatrix, nx::Integer, ny::Integer, margin_fraction::Real)
    xlo, xhi = ScoreEstimation.determine_value_range(samples[1:1, :])
    ylo, yhi = ScoreEstimation.determine_value_range(samples[2:2, :])
    xpad = max(Float64(margin_fraction) * (xhi - xlo), 1.0e-3)
    ypad = max(Float64(margin_fraction) * (yhi - ylo), 1.0e-3)
    return (
        x=collect(range(xlo - xpad, xhi + xpad; length=Int(nx))),
        y=collect(range(ylo - ypad, yhi + ypad; length=Int(ny))),
    )
end

function finite_difference_axis(field::AbstractMatrix, grid::AbstractVector{<:Real}, axis::Integer)
    out = Matrix{Float32}(undef, size(field))
    if axis == 1
        @views out[1, :] .= (field[2, :] .- field[1, :]) ./ Float32(grid[2] - grid[1])
        @views out[end, :] .= (field[end, :] .- field[end - 1, :]) ./ Float32(grid[end] - grid[end - 1])
        for i in 2:(size(field, 1) - 1)
            @views out[i, :] .= (field[i + 1, :] .- field[i - 1, :]) ./ Float32(grid[i + 1] - grid[i - 1])
        end
    else
        @views out[:, 1] .= (field[:, 2] .- field[:, 1]) ./ Float32(grid[2] - grid[1])
        @views out[:, end] .= (field[:, end] .- field[:, end - 1]) ./ Float32(grid[end] - grid[end - 1])
        for j in 2:(size(field, 2) - 1)
            @views out[:, j] .= (field[:, j + 1] .- field[:, j - 1]) ./ Float32(grid[j + 1] - grid[j - 1])
        end
    end
    return out
end

function mobility_feature_statistics(samples::AbstractMatrix,
                                     score_wrapper,
                                     state_mean::AbstractVector,
                                     state_scale::AbstractVector;
                                     sample_count::Integer=50_000,
                                     batch_size::Integer=8192,
                                     seed::Integer=0)
    nsamples = size(samples, 2)
    take = min(Int(sample_count), nsamples)
    rng = MersenneTwister(seed)
    indices = take < nsamples ? sort(rand(rng, 1:nsamples, take)) : collect(1:nsamples)
    subset = Float32.(samples[:, indices])
    score_vals = batched_model_outputs(score_wrapper, subset; batch_size=batch_size, use_gpu=false)
    μ = reshape(Float32.(state_mean), :, 1)
    σ = reshape(Float32.(state_scale), :, 1)
    xnorm = (subset .- μ) ./ σ
    xc = subset .- Float32.(state_mean)
    features = vcat(
        xnorm,
        Float32.(score_vals),
        reshape(@view(xc[1, :]) .^ 2, 1, :),
        reshape(@view(xc[1, :]) .* @view(xc[2, :]), 1, :),
        reshape(@view(xc[2, :]) .^ 2, 1, :),
    )
    feature_mean = vec(mean(features; dims=2))
    feature_scale = vec(std(features; dims=2))
    feature_scale .= max.(feature_scale, 1.0f-5)
    feature_mean[1:2] .= 0f0
    feature_scale[1:2] .= 1f0
    return feature_mean, feature_scale
end

function percentile_field_axes(samples::AbstractMatrix,
                               nx::Integer,
                               ny::Integer;
                               lower_quantile::Real=0.01,
                               upper_quantile::Real=0.99,
                               padding_fraction::Real=0.05)
    xs = Float64.(vec(samples[1, :]))
    ys = Float64.(vec(samples[2, :]))
    xlo = quantile(xs, Float64(lower_quantile))
    xhi = quantile(xs, Float64(upper_quantile))
    ylo = quantile(ys, Float64(lower_quantile))
    yhi = quantile(ys, Float64(upper_quantile))
    xpad = max(Float64(padding_fraction) * (xhi - xlo), 1.0e-3)
    ypad = max(Float64(padding_fraction) * (yhi - ylo), 1.0e-3)
    return (
        x=collect(range(xlo - xpad, xhi + xpad; length=Int(nx))),
        y=collect(range(ylo - ypad, yhi + ypad; length=Int(ny))),
    )
end

function batched_mobility_components(model::DeltaMMobilityField,
                                     score_wrapper,
                                     points::AbstractMatrix;
                                     batch_size::Integer=8192,
                                     use_gpu::Bool=false)
    n_points = size(points, 2)
    fields = Dict(
        :sigma11 => Vector{Float32}(undef, n_points),
        :sigma21 => Vector{Float32}(undef, n_points),
        :sigma22 => Vector{Float32}(undef, n_points),
        :omega => Vector{Float32}(undef, n_points),
        :m11 => Vector{Float32}(undef, n_points),
        :m12 => Vector{Float32}(undef, n_points),
        :m21 => Vector{Float32}(undef, n_points),
        :m22 => Vector{Float32}(undef, n_points),
        :delta11 => Vector{Float32}(undef, n_points),
        :delta12 => Vector{Float32}(undef, n_points),
        :delta21 => Vector{Float32}(undef, n_points),
        :delta22 => Vector{Float32}(undef, n_points),
    )
    model_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(model), CUDA)
    else
        deepcopy(model)
    end
    score_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(score_wrapper), CUDA)
    else
        deepcopy(score_wrapper)
    end
    Flux.testmode!(model_eval)
    Flux.testmode!(score_eval)

    start_idx = 1
    while start_idx <= n_points
        stop_idx = min(start_idx + batch_size - 1, n_points)
        batch = ScoreEstimation._to_device(Float32.(points[:, start_idx:stop_idx]); use_gpu=use_gpu)
        score_batch = score_eval(batch)
        comps = mobility_components(model_eval, batch, score_batch)
        for key in keys(fields)
            @views fields[key][start_idx:stop_idx] .= Array(getfield(comps, key))
        end
        start_idx = stop_idx + 1
    end

    return fields
end

function build_delta_m_grids(model::DeltaMMobilityField,
                             score_wrapper,
                             axes_info;
                             batch_size::Integer=8192,
                             use_gpu::Bool=false,
                             divergence_mode::Symbol=:finite_difference)
    xgrid = axes_info.x
    ygrid = axes_info.y
    nx = length(xgrid)
    ny = length(ygrid)
    points = Matrix{Float32}(undef, 2, nx * ny)
    idx = 1
    for iy in eachindex(ygrid), ix in eachindex(xgrid)
        points[1, idx] = Float32(xgrid[ix])
        points[2, idx] = Float32(ygrid[iy])
        idx += 1
    end

    if divergence_mode === :autodiff
        fields = batched_mobility_components(model, score_wrapper, points; batch_size=batch_size, use_gpu=false)
        score = batched_model_outputs(score_wrapper, points; batch_size=batch_size, use_gpu=false)
        reshape_field_autodiff(values) = reshape(Float32.(values), nx, ny)
        m11 = reshape_field_autodiff(fields[:m11])
        m12 = reshape_field_autodiff(fields[:m12])
        m21 = reshape_field_autodiff(fields[:m21])
        m22 = reshape_field_autodiff(fields[:m22])
        sigma11 = reshape_field_autodiff(fields[:sigma11])
        sigma21 = reshape_field_autodiff(fields[:sigma21])
        sigma22 = reshape_field_autodiff(fields[:sigma22])
        delta11 = reshape_field_autodiff(fields[:delta11])
        delta12 = reshape_field_autodiff(fields[:delta12])
        delta21 = reshape_field_autodiff(fields[:delta21])
        delta22 = reshape_field_autodiff(fields[:delta22])
        omega = reshape_field_autodiff(fields[:omega])
        score1 = reshape(Float32.(score[1, :]), nx, ny)
        score2 = reshape(Float32.(score[2, :]), nx, ny)

        drift_grid = Array{Float32}(undef, 2, nx, ny)
        score_term_grid = Array{Float32}(undef, 2, nx, ny)
        divergence_grid = Array{Float32}(undef, 2, nx, ny)
        sigma_grid = Array{Float32}(undef, 2, 2, nx, ny)
        delta_grid = Array{Float32}(undef, 2, 2, nx, ny)
        lambda_min = Array{Float32}(undef, nx, ny)

        for iy in eachindex(ygrid), ix in eachindex(xgrid)
            divergence = autodiff_mobility_divergence(model, score_wrapper, Float64[xgrid[ix], ygrid[iy]])
            score_term = Float64[
                m11[ix, iy] * score1[ix, iy] + m12[ix, iy] * score2[ix, iy],
                m21[ix, iy] * score1[ix, iy] + m22[ix, iy] * score2[ix, iy],
            ]
            drift_grid[:, ix, iy] .= Float32.(score_term .+ divergence)
            score_term_grid[:, ix, iy] .= Float32.(score_term)
            divergence_grid[:, ix, iy] .= Float32.(divergence)
            sigma_grid[1, 1, ix, iy] = sigma11[ix, iy]
            sigma_grid[1, 2, ix, iy] = 0f0
            sigma_grid[2, 1, ix, iy] = sigma21[ix, iy]
            sigma_grid[2, 2, ix, iy] = sigma22[ix, iy]
            delta_grid[1, 1, ix, iy] = delta11[ix, iy]
            delta_grid[1, 2, ix, iy] = delta12[ix, iy]
            delta_grid[2, 1, ix, iy] = delta21[ix, iy]
            delta_grid[2, 2, ix, iy] = delta22[ix, iy]
            lambda_min[ix, iy] = 0.5f0 * ((m11[ix, iy] + m22[ix, iy]) - sqrt(max((m11[ix, iy] - m22[ix, iy])^2 + 4f0 * ((0.5f0 * (m12[ix, iy] + m21[ix, iy]))^2), 0f0)))
        end

        return (
            xgrid=xgrid,
            ygrid=ygrid,
            drift_grid=drift_grid,
            score_term_grid=score_term_grid,
            divergence_grid=divergence_grid,
            sigma_grid=sigma_grid,
            delta_grid=delta_grid,
            omega=omega,
            lambda_min=lambda_min,
            divergence_mode=String(divergence_mode),
        )
    end

    divergence_mode === :finite_difference || error("Unsupported divergence mode $(divergence_mode).")
    fields = batched_mobility_components(model, score_wrapper, points; batch_size=batch_size, use_gpu=use_gpu)
    score = batched_model_outputs(score_wrapper, points; batch_size=batch_size, use_gpu=use_gpu)
    reshape_field_fd(values) = reshape(Float32.(values), nx, ny)
    m11 = reshape_field_fd(fields[:m11])
    m12 = reshape_field_fd(fields[:m12])
    m21 = reshape_field_fd(fields[:m21])
    m22 = reshape_field_fd(fields[:m22])
    delta11 = reshape_field_fd(fields[:delta11])
    delta12 = reshape_field_fd(fields[:delta12])
    delta21 = reshape_field_fd(fields[:delta21])
    delta22 = reshape_field_fd(fields[:delta22])
    sigma11 = reshape_field_fd(fields[:sigma11])
    sigma21 = reshape_field_fd(fields[:sigma21])
    sigma22 = reshape_field_fd(fields[:sigma22])
    omega = reshape_field_fd(fields[:omega])
    score1 = reshape(Float32.(score[1, :]), nx, ny)
    score2 = reshape(Float32.(score[2, :]), nx, ny)

    div1 = finite_difference_axis(m11, xgrid, 1) .+ finite_difference_axis(m12, ygrid, 2)
    div2 = finite_difference_axis(m21, xgrid, 1) .+ finite_difference_axis(m22, ygrid, 2)
    score_term1 = m11 .* score1 .+ m12 .* score2
    score_term2 = m21 .* score1 .+ m22 .* score2
    drift1 = score_term1 .+ div1
    drift2 = score_term2 .+ div2
    lambda_min = 0.5f0 .* ((m11 .+ m22) .- sqrt.(max.((m11 .- m22) .^ 2 .+ 4f0 .* ((0.5f0 .* (m12 .+ m21)) .^ 2), 0f0)))

    drift_grid = Array{Float32}(undef, 2, nx, ny)
    score_term_grid = Array{Float32}(undef, 2, nx, ny)
    divergence_grid = Array{Float32}(undef, 2, nx, ny)
    sigma_grid = Array{Float32}(undef, 2, 2, nx, ny)
    delta_grid = Array{Float32}(undef, 2, 2, nx, ny)
    drift_grid[1, :, :] .= drift1
    drift_grid[2, :, :] .= drift2
    score_term_grid[1, :, :] .= score_term1
    score_term_grid[2, :, :] .= score_term2
    divergence_grid[1, :, :] .= div1
    divergence_grid[2, :, :] .= div2
    sigma_grid[1, 1, :, :] .= sigma11
    sigma_grid[1, 2, :, :] .= 0f0
    sigma_grid[2, 1, :, :] .= sigma21
    sigma_grid[2, 2, :, :] .= sigma22
    delta_grid[1, 1, :, :] .= delta11
    delta_grid[1, 2, :, :] .= delta12
    delta_grid[2, 1, :, :] .= delta21
    delta_grid[2, 2, :, :] .= delta22

    return (
        xgrid=xgrid,
        ygrid=ygrid,
        drift_grid=drift_grid,
        score_term_grid=score_term_grid,
        divergence_grid=divergence_grid,
        sigma_grid=sigma_grid,
        delta_grid=delta_grid,
        omega=omega,
        lambda_min=lambda_min,
        divergence_mode=String(divergence_mode),
    )
end

function read_h5_trajectory(path::AbstractString)
    return h5open(path, "r") do file
        (
            stationary=Float32.(read(file["stationary_samples"])),
            trajectory=Float32.(read(file["trajectory"])),
        )
    end
end

function cphi_figure_rows(fig, start_row, result, lag_times; candidate_label="δM", baseline=nothing)
    output_dim = length(result.output_labels)
    Label(fig[start_row, 1:3], result.name, fontsize=24, font=:bold)
    row0 = start_row + 1
    for output_idx in 1:output_dim
        for state_idx in 1:2
            ax = Axis(
                fig[row0 + output_idx - 1, state_idx],
                xlabel=output_idx == output_dim ? "t" : "",
                ylabel=state_idx == 1 ? result.output_labels[output_idx] : "",
                title="[$(output_idx),$(state_idx)]",
            )
            lines!(ax, lag_times, vec(result.C_obs[output_idx, state_idx, :]); color=OBS_COLOR, linewidth=2.5, label="obs")
            baseline === nothing || lines!(ax, lag_times, vec(baseline.C_model[output_idx, state_idx, :]); color=:gray50, linewidth=2.0, label="baseline")
            lines!(ax, lag_times, vec(result.C_model[output_idx, state_idx, :]); color=METHOD_COLOR, linewidth=2.0, label=candidate_label)
            axislegend(ax, position=:rt)
        end
    end
    ax_norm = Axis(fig[row0:(row0 + output_dim - 1), 3], xlabel="t", ylabel="rel. Fro err", title="$(result.name) error")
    lines!(ax_norm, lag_times, result.relative_error; color=METHOD_COLOR, linewidth=2.5, label=candidate_label)
    baseline === nothing || lines!(ax_norm, lag_times, baseline.relative_error; color=:gray50, linewidth=2.0, label="baseline")
    axislegend(ax_norm, position=:rt)
    return row0 + output_dim
end

function delta_m_training_figure(losses,
                                 mean_penalties,
                                 magnitude_penalties,
                                 grid_data,
                                 observable_results,
                                 lag_times)
    total_rows = 8 + sum(length(result.output_labels) + 1 for result in observable_results)
    fig = Figure(size=(2400, 300 * total_rows))
    ax_loss = Axis(fig[1, 1:2], xlabel="epoch", ylabel="loss", title="δM training loss")
    lines!(ax_loss, 1:length(losses), losses; color=METHOD_COLOR, linewidth=2.5)
    ax_mean = Axis(fig[1, 3], xlabel="epoch", ylabel="penalty", title="Mean-zero penalty")
    lines!(ax_mean, 1:length(mean_penalties), mean_penalties; color=:darkorange2, linewidth=2.5)
    ax_mag = Axis(fig[1, 4], xlabel="epoch", ylabel="penalty", title="Magnitude penalty")
    lines!(ax_mag, 1:length(magnitude_penalties), magnitude_penalties; color=:seagreen4, linewidth=2.5)

    titles = ("δM₁₁", "δM₁₂", "δM₂₁", "δM₂₂")
    for idx in 1:4
        ax = Axis(fig[2 + div(idx - 1, 2), 1 + mod(idx - 1, 2)], xlabel="x", ylabel="y", title=titles[idx])
        heatmap!(ax, grid_data.xgrid, grid_data.ygrid, permutedims(grid_data.delta_grid[div(idx - 1, 2) + 1, mod(idx - 1, 2) + 1, :, :]); colormap=:balance)
    end
    ax_omega = Axis(fig[2, 3], xlabel="x", ylabel="y", title="ω(x)")
    heatmap!(ax_omega, grid_data.xgrid, grid_data.ygrid, permutedims(grid_data.omega); colormap=:balance)
    ax_lambda = Axis(fig[2, 4], xlabel="x", ylabel="y", title="λ_min(D)")
    heatmap!(ax_lambda, grid_data.xgrid, grid_data.ygrid, permutedims(grid_data.lambda_min); colormap=:viridis)

    row = 4
    for result in observable_results
        Label(fig[row, 1:3], result.name, fontsize=24, font=:bold)
        row += 1
        output_dim = length(result.output_labels)
        for output_idx in 1:output_dim
            for state_idx in 1:2
                ax = Axis(
                    fig[row + output_idx - 1, state_idx],
                    xlabel=output_idx == output_dim ? "t" : "",
                    ylabel=state_idx == 1 ? result.output_labels[output_idx] : "",
                    title="E[$(output_idx),$(state_idx)]",
                )
                lines!(ax, lag_times, vec(result.E_obs[output_idx, state_idx, :]); color=OBS_COLOR, linewidth=2.5, label="obs")
                lines!(ax, lag_times, vec(result.E_model[output_idx, state_idx, :]); color=METHOD_COLOR, linewidth=2.0, label="δM")
                axislegend(ax, position=:rt)
            end
        end
        ax_err = Axis(fig[row:(row + output_dim - 1), 3], xlabel="t", ylabel="||ΔE||_F / ||E_obs||_F", title="$(result.name) residual error")
        lines!(ax_err, lag_times, result.relative_error; color=:darkorange2, linewidth=2.5)
        row += output_dim
    end

    Label(fig[0, :], "Brusselator δM Training Diagnostics", fontsize=28)
    return fig
end

function delta_m_rollout_figure(observable_results,
                                lag_times,
                                true_diag,
                                baseline_diag,
                                model_diag,
                                summary_lines)
    total_rows = 3 + sum(length(result.output_labels) + 1 for result in observable_results)
    fig = Figure(size=(2500, 280 * total_rows))
    row = 1
    for result in observable_results
        row = cphi_figure_rows(fig, row, result, lag_times; baseline=result.baseline)
    end

    ax_pdf_x = Axis(fig[row, 1], xlabel="x", ylabel="density", title="PDF x")
    ax_pdf_y = Axis(fig[row, 2], xlabel="y", ylabel="density", title="PDF y")
    ax_acf_x = Axis(fig[row, 3], xlabel="lag", ylabel="ACF", title="ACF x")
    ax_acf_y = Axis(fig[row, 4], xlabel="lag", ylabel="ACF", title="ACF y")
    lines!(ax_pdf_x, true_diag.pdf_x.x, true_diag.pdf_x.density; color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_pdf_x, baseline_diag.pdf_x.x, baseline_diag.pdf_x.density; color=:gray50, linewidth=2.0, label="baseline")
    lines!(ax_pdf_x, model_diag.pdf_x.x, model_diag.pdf_x.density; color=METHOD_COLOR, linewidth=2.0, label="δM")
    lines!(ax_pdf_y, true_diag.pdf_y.x, true_diag.pdf_y.density; color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_pdf_y, baseline_diag.pdf_y.x, baseline_diag.pdf_y.density; color=:gray50, linewidth=2.0, label="baseline")
    lines!(ax_pdf_y, model_diag.pdf_y.x, model_diag.pdf_y.density; color=METHOD_COLOR, linewidth=2.0, label="δM")
    lines!(ax_acf_x, true_diag.lag_axis, true_diag.acf_x; color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_acf_x, true_diag.lag_axis, resample_to_axis(baseline_diag.lag_axis, baseline_diag.acf_x, true_diag.lag_axis); color=:gray50, linewidth=2.0, label="baseline")
    lines!(ax_acf_x, true_diag.lag_axis, resample_to_axis(model_diag.lag_axis, model_diag.acf_x, true_diag.lag_axis); color=METHOD_COLOR, linewidth=2.0, label="δM")
    lines!(ax_acf_y, true_diag.lag_axis, true_diag.acf_y; color=OBS_COLOR, linewidth=2.5, label="obs")
    lines!(ax_acf_y, true_diag.lag_axis, resample_to_axis(baseline_diag.lag_axis, baseline_diag.acf_y, true_diag.lag_axis); color=:gray50, linewidth=2.0, label="baseline")
    lines!(ax_acf_y, true_diag.lag_axis, resample_to_axis(model_diag.lag_axis, model_diag.acf_y, true_diag.lag_axis); color=METHOD_COLOR, linewidth=2.0, label="δM")
    for ax in (ax_pdf_x, ax_pdf_y, ax_acf_x, ax_acf_y)
        axislegend(ax, position=:rt)
    end

    summary_ax = Axis(fig[row + 1, 1:4], title="Summary")
    hidedecorations!(summary_ax)
    hidespines!(summary_ax)
    text!(summary_ax, 0.0, 1.0; text=join(summary_lines, "\n"), space=:relative, align=(:left, :top), fontsize=22)

    Label(fig[0, :], "Brusselator δM Rollout Validation", fontsize=28)
    return fig
end

function grid_edges(centers::AbstractVector{<:Real})
    n = length(centers)
    edges = Vector{Float64}(undef, n + 1)
    for i in 2:n
        edges[i] = 0.5 * (Float64(centers[i - 1]) + Float64(centers[i]))
    end
    edges[1] = Float64(centers[1]) - (edges[2] - Float64(centers[1]))
    edges[end] = Float64(centers[end]) + (Float64(centers[end]) - edges[end - 1])
    return edges
end

function stationary_density_grid(samples::AbstractMatrix,
                                 xgrid::AbstractVector{<:Real},
                                 ygrid::AbstractVector{<:Real})
    nx = length(xgrid)
    ny = length(ygrid)
    xedges = grid_edges(xgrid)
    yedges = grid_edges(ygrid)
    counts = zeros(Float64, nx, ny)
    for idx in axes(samples, 2)
        x = Float64(samples[1, idx])
        y = Float64(samples[2, idx])
        ix = searchsortedlast(xedges, x) - 1
        iy = searchsortedlast(yedges, y) - 1
        if 1 <= ix <= nx && 1 <= iy <= ny
            counts[ix, iy] += 1
        end
    end
    dx = mean(diff(Float64.(xgrid)))
    dy = mean(diff(Float64.(ygrid)))
    density = counts ./ max(sum(counts) * dx * dy, eps(Float64))
    weights = counts ./ max(sum(counts), eps(Float64))
    return density, weights
end

function true_brusselator_fields(cfg::BrusselatorConfig,
                                 xgrid::AbstractVector{<:Real},
                                 ygrid::AbstractVector{<:Real})
    nx = length(xgrid)
    ny = length(ygrid)
    drift = Array{Float32}(undef, 2, nx, ny)
    diffusion = Array{Float32}(undef, 2, 2, nx, ny)
    du = zeros(Float64, 2)
    for iy in eachindex(ygrid), ix in eachindex(xgrid)
        u = Float64[xgrid[ix], ygrid[iy]]
        brusselator_drift!(du, u, cfg, 0.0)
        Σ = brusselator_sigma(u, cfg, 0.0)
        G = Σ * transpose(Σ)
        drift[1, ix, iy] = Float32(du[1])
        drift[2, ix, iy] = Float32(du[2])
        diffusion[:, :, ix, iy] .= Float32.(0.5 .* G)
    end
    return drift, diffusion
end

function baseline_rom_fields(score_wrapper,
                             Phi::AbstractMatrix,
                             Sigma::AbstractMatrix,
                             xgrid::AbstractVector{<:Real},
                             ygrid::AbstractVector{<:Real})
    nx = length(xgrid)
    ny = length(ygrid)
    points = Matrix{Float32}(undef, 2, nx * ny)
    idx = 1
    for iy in eachindex(ygrid), ix in eachindex(xgrid)
        points[1, idx] = Float32(xgrid[ix])
        points[2, idx] = Float32(ygrid[iy])
        idx += 1
    end
    score = batched_model_outputs(score_wrapper, points; batch_size=8192, use_gpu=false)
    drift_flat = Float32.(Phi) * Float32.(score)
    drift = reshape(Float32.(drift_flat), 2, nx, ny)
    diffusion = Array{Float32}(undef, 2, 2, nx, ny)
    G = Float32.(Sigma * transpose(Sigma))
    for iy in eachindex(ygrid), ix in eachindex(xgrid)
        diffusion[:, :, ix, iy] .= G
    end
    return drift, diffusion
end

function sigma_grid_to_diffusion_tensor(sigma_grid::AbstractArray{<:Real,4})
    nx = size(sigma_grid, 3)
    ny = size(sigma_grid, 4)
    G = Array{Float32}(undef, 2, 2, nx, ny)
    for iy in 1:ny, ix in 1:nx
        Σ = Float32.(sigma_grid[:, :, ix, iy])
        G[:, :, ix, iy] .= Σ * transpose(Σ)
    end
    return G
end

function weighted_rmse(error_field::AbstractMatrix, weights::AbstractMatrix)
    return sqrt(sum(weights .* (Float64.(error_field) .^ 2)) / max(sum(weights), eps(Float64)))
end

function coefficient_metrics(truth_drift,
                             truth_diffusion,
                             base_drift,
                             base_diffusion,
                             model_drift,
                             model_diffusion,
                             weights)
    components = Dict(
        "b1" => (Float64.(truth_drift[1, :, :]), Float64.(base_drift[1, :, :]), Float64.(model_drift[1, :, :])),
        "b2" => (Float64.(truth_drift[2, :, :]), Float64.(base_drift[2, :, :]), Float64.(model_drift[2, :, :])),
        "G11" => (Float64.(truth_diffusion[1, 1, :, :]), Float64.(base_diffusion[1, 1, :, :]), Float64.(model_diffusion[1, 1, :, :])),
        "G12" => (Float64.(truth_diffusion[1, 2, :, :]), Float64.(base_diffusion[1, 2, :, :]), Float64.(model_diffusion[1, 2, :, :])),
        "G22" => (Float64.(truth_diffusion[2, 2, :, :]), Float64.(base_diffusion[2, 2, :, :]), Float64.(model_diffusion[2, 2, :, :])),
    )
    metrics = Dict{String,Any}()
    for (name, (truth, base, model)) in components
        base_err = base .- truth
        model_err = model .- truth
        metrics[name] = (
            full_baseline_rmse=sqrt(mean(abs2, base_err)),
            full_model_rmse=sqrt(mean(abs2, model_err)),
            weighted_baseline_rmse=weighted_rmse(base_err, weights),
            weighted_model_rmse=weighted_rmse(model_err, weights),
        )
    end
    drift_base_norm = sqrt.(sum((Float64.(base_drift) .- Float64.(truth_drift)) .^ 2; dims=1))[1, :, :]
    drift_model_norm = sqrt.(sum((Float64.(model_drift) .- Float64.(truth_drift)) .^ 2; dims=1))[1, :, :]
    diff_base_norm = sqrt.(sum((Float64.(base_diffusion) .- Float64.(truth_diffusion)) .^ 2; dims=(1, 2)))[1, 1, :, :]
    diff_model_norm = sqrt.(sum((Float64.(model_diffusion) .- Float64.(truth_diffusion)) .^ 2; dims=(1, 2)))[1, 1, :, :]
    return metrics, drift_base_norm, drift_model_norm, diff_base_norm, diff_model_norm
end

function coefficient_summary_text(metrics::Dict{String,Any}, prefix::String)
    order = ["b1", "b2", "G11", "G12", "G22"]
    lines = String[]
    for name in order
        vals = metrics[name]
        push!(lines, "$(prefix) $(name): RMSE=$(round(vals.full_model_rmse, digits=4))")
    end
    return join(lines, "\n")
end

function delta_m_coefficients_figure(xgrid,
                                     ygrid,
                                     density,
                                     truth_drift,
                                     truth_diffusion,
                                     base_drift,
                                     base_diffusion,
                                     model_drift,
                                     model_diffusion,
                                     metrics,
                                     drift_base_norm,
                                     drift_model_norm,
                                     diff_base_norm,
                                     diff_model_norm,
                                     summary_lines)
    fig = Figure(size=(3600, 4600))
    ax_density = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Observed stationary density")
    heatmap!(ax_density, xgrid, ygrid, permutedims(density); colormap=:viridis)
    ax_drift_summary = Axis(fig[1, 2], title="Drift metrics")
    hidedecorations!(ax_drift_summary); hidespines!(ax_drift_summary)
    drift_lines = ["b1: $(round(metrics["b1"].weighted_model_rmse, digits=4)) vs $(round(metrics["b1"].weighted_baseline_rmse, digits=4))",
                   "b2: $(round(metrics["b2"].weighted_model_rmse, digits=4)) vs $(round(metrics["b2"].weighted_baseline_rmse, digits=4))"]
    text!(ax_drift_summary, 0, 1; text="weighted RMSE\nδM vs baseline\n" * join(drift_lines, "\n"), space=:relative, align=(:left, :top), fontsize=20)
    ax_diff_summary = Axis(fig[1, 3], title="Diffusion metrics")
    hidedecorations!(ax_diff_summary); hidespines!(ax_diff_summary)
    diff_lines = ["G11: $(round(metrics["G11"].weighted_model_rmse, digits=4)) vs $(round(metrics["G11"].weighted_baseline_rmse, digits=4))",
                  "G12: $(round(metrics["G12"].weighted_model_rmse, digits=4)) vs $(round(metrics["G12"].weighted_baseline_rmse, digits=4))",
                  "G22: $(round(metrics["G22"].weighted_model_rmse, digits=4)) vs $(round(metrics["G22"].weighted_baseline_rmse, digits=4))"]
    text!(ax_diff_summary, 0, 1; text="weighted RMSE\nδM vs baseline\n" * join(diff_lines, "\n"), space=:relative, align=(:left, :top), fontsize=20)
    ax_drift_scatter = Axis(fig[1, 4], xlabel="true drift", ylabel="predicted drift", title="Drift true-vs-model")
    drift_truth_vec = vcat(vec(Float64.(truth_drift[1, :, :])), vec(Float64.(truth_drift[2, :, :])))
    scatter!(ax_drift_scatter, drift_truth_vec, vcat(vec(Float64.(base_drift[1, :, :])), vec(Float64.(base_drift[2, :, :]))); color=(:gray40, 0.35), markersize=4, label="baseline")
    scatter!(ax_drift_scatter, drift_truth_vec, vcat(vec(Float64.(model_drift[1, :, :])), vec(Float64.(model_drift[2, :, :]))); color=(Makie.wong_colors()[1], 0.35), markersize=4, label="δM")
    axislegend(ax_drift_scatter, position=:lt)
    ax_diff_scatter = Axis(fig[1, 5], xlabel="true diffusion", ylabel="predicted diffusion", title="Diffusion true-vs-model")
    diff_truth_vec = vcat(vec(Float64.(truth_diffusion[1, 1, :, :])), vec(Float64.(truth_diffusion[1, 2, :, :])), vec(Float64.(truth_diffusion[2, 2, :, :])))
    scatter!(ax_diff_scatter, diff_truth_vec, vcat(vec(Float64.(base_diffusion[1, 1, :, :])), vec(Float64.(base_diffusion[1, 2, :, :])), vec(Float64.(base_diffusion[2, 2, :, :]))); color=(:gray40, 0.35), markersize=4, label="baseline")
    scatter!(ax_diff_scatter, diff_truth_vec, vcat(vec(Float64.(model_diffusion[1, 1, :, :])), vec(Float64.(model_diffusion[1, 2, :, :])), vec(Float64.(model_diffusion[2, 2, :, :]))); color=(Makie.wong_colors()[1], 0.35), markersize=4, label="δM")
    axislegend(ax_diff_scatter, position=:lt)

    row_specs = [
        ("b1", Float64.(truth_drift[1, :, :]), Float64.(base_drift[1, :, :]), Float64.(model_drift[1, :, :])),
        ("b2", Float64.(truth_drift[2, :, :]), Float64.(base_drift[2, :, :]), Float64.(model_drift[2, :, :])),
        ("G11", Float64.(truth_diffusion[1, 1, :, :]), Float64.(base_diffusion[1, 1, :, :]), Float64.(model_diffusion[1, 1, :, :])),
        ("G12", Float64.(truth_diffusion[1, 2, :, :]), Float64.(base_diffusion[1, 2, :, :]), Float64.(model_diffusion[1, 2, :, :])),
        ("G22", Float64.(truth_diffusion[2, 2, :, :]), Float64.(base_diffusion[2, 2, :, :]), Float64.(model_diffusion[2, 2, :, :])),
    ]
    row_idx = 2
    for (name, truth, base, model) in row_specs
        bound = maximum(abs, vcat(vec(truth), vec(base), vec(model)))
        err_base = abs.(base .- truth)
        err_model = abs.(model .- truth)
        err_bound = maximum(vcat(vec(err_base), vec(err_model)))
        titles = ("true", "baseline", "δM", "baseline error", "δM error")
        fields = (truth, base, model, err_base, err_model)
        for col_idx in 1:5
            ax = Axis(fig[row_idx, col_idx], xlabel="x", ylabel=col_idx == 1 ? "y" : "", title=col_idx == 1 ? "$(name) $(titles[col_idx])" : titles[col_idx])
            if col_idx <= 3
                heatmap!(ax, xgrid, ygrid, permutedims(fields[col_idx]); colormap=:balance, colorrange=(-bound, bound))
            else
                heatmap!(ax, xgrid, ygrid, permutedims(fields[col_idx]); colormap=:viridis, colorrange=(0, err_bound))
            end
        end
        row_idx += 1
    end

    drift_norm_gl = GridLayout(fig[7, 1])
    diff_norm_gl = GridLayout(fig[7, 2])
    ax_drift_norm_base = Axis(drift_norm_gl[1, 1], xlabel="x", ylabel="y", title="Drift norm err baseline")
    ax_drift_norm_model = Axis(drift_norm_gl[1, 2], xlabel="x", ylabel="", title="Drift norm err δM")
    db_bound = maximum(vcat(vec(drift_base_norm), vec(drift_model_norm)))
    heatmap!(ax_drift_norm_base, xgrid, ygrid, permutedims(drift_base_norm); colormap=:viridis, colorrange=(0, db_bound))
    heatmap!(ax_drift_norm_model, xgrid, ygrid, permutedims(drift_model_norm); colormap=:viridis, colorrange=(0, db_bound))
    ax_diff_norm_base = Axis(diff_norm_gl[1, 1], xlabel="x", ylabel="y", title="Diffusion norm err baseline")
    ax_diff_norm_model = Axis(diff_norm_gl[1, 2], xlabel="x", ylabel="", title="Diffusion norm err δM")
    dg_bound = maximum(vcat(vec(diff_base_norm), vec(diff_model_norm)))
    heatmap!(ax_diff_norm_base, xgrid, ygrid, permutedims(diff_base_norm); colormap=:viridis, colorrange=(0, dg_bound))
    heatmap!(ax_diff_norm_model, xgrid, ygrid, permutedims(diff_model_norm); colormap=:viridis, colorrange=(0, dg_bound))

    ax_bar = Axis(fig[7, 3], ylabel="RMSE", title="Component RMSE")
    labels = ["b1", "b2", "G11", "G12", "G22"]
    base_vals = [metrics[label].full_baseline_rmse for label in labels]
    model_vals = [metrics[label].full_model_rmse for label in labels]
    xs = 1:length(labels)
    barplot!(ax_bar, xs .- 0.18, base_vals; width=0.32, color=:gray50, label="baseline")
    barplot!(ax_bar, xs .+ 0.18, model_vals; width=0.32, color=METHOD_COLOR, label="δM")
    ax_bar.xticks = (xs, labels)
    axislegend(ax_bar, position=:rt)
    ax_wbar = Axis(fig[7, 4], ylabel="weighted RMSE", title="Density-weighted RMSE")
    base_wvals = [metrics[label].weighted_baseline_rmse for label in labels]
    model_wvals = [metrics[label].weighted_model_rmse for label in labels]
    barplot!(ax_wbar, xs .- 0.18, base_wvals; width=0.32, color=:gray50, label="baseline")
    barplot!(ax_wbar, xs .+ 0.18, model_wvals; width=0.32, color=METHOD_COLOR, label="δM")
    ax_wbar.xticks = (xs, labels)
    axislegend(ax_wbar, position=:rt)
    ax_summary = Axis(fig[7, 5], title="Summary")
    hidedecorations!(ax_summary); hidespines!(ax_summary)
    text!(ax_summary, 0, 1; text=join(summary_lines, "\n"), space=:relative, align=(:left, :top), fontsize=18)

    Label(fig[0, :], "Brusselator True-vs-ROM Coefficient Comparison", fontsize=30)
    return fig
end

function primary_artifact(paths, stage::Symbol)
    stage === :generate_data && return paths.dataset_h5
    stage === :train_score && return paths.score_model_bson
    stage === :estimate_phi_sigma && return paths.phi_sigma_h5
    stage === :observed_residuals && return paths.observed_residuals_png
    stage === :overview && return paths.overview_png
    stage === :joint_score && return paths.joint_score_png
    stage === :delta_m && return paths.delta_m_png
    stage === :delta_m_rollout && return paths.delta_m_rollout_png
    stage === :delta_m_coefficients && return paths.delta_m_coefficients_png
    error("Unsupported stage `$(stage)`.")
end

function should_run_stage(paths, stage::Symbol, force::Bool)
    return force || !isfile(primary_artifact(paths, stage))
end

function run_generate_data(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :generate_data, force)
        return Dict("status" => "skipped", "dataset_h5" => paths.dataset_h5)
    end

    cfg = build_generation_config(config)
    Random.seed!(cfg.seed)
    correlated, time_axis = integrate_brusselator_timeseries(cfg)
    stride, decorrelation_times = estimate_uncorrelated_stride(correlated)
    uncorrelated = integrate_brusselator_stationary(cfg; flatten=true)

    h5open(paths.dataset_h5, "w") do file
        file["uncorrelated_samples"] = uncorrelated
        file["correlated_samples"] = correlated
        file["time_axis"] = time_axis
        file["decorrelation_times"] = decorrelation_times
        file["decorrelation_stride"] = stride
        file["A"] = cfg.A
        file["B"] = cfg.B
        file["Omega"] = cfg.Ω
        file["dt"] = cfg.dt
        file["burnin_steps"] = cfg.burnin_steps
        file["n_steps_timeseries"] = cfg.n_steps_timeseries
        file["resolution_timeseries"] = cfg.resolution_timeseries
        file["n_steps_stationary"] = cfg.n_steps_stationary
        file["burnin_stationary"] = cfg.burnin_stationary
        file["resolution_stationary"] = cfg.resolution_stationary
        file["ensemble_stationary"] = cfg.ensemble_stationary
        file["maxlag"] = cfg.maxlag
        file["plot_window"] = cfg.plot_window
        file["phase_points"] = cfg.phase_points
        file["seed"] = cfg.seed
    end

    return Dict(
        "status" => "completed",
        "dataset_h5" => paths.dataset_h5,
        "uncorrelated_shape" => Int[size(uncorrelated, 1), size(uncorrelated, 2)],
        "correlated_shape" => Int[size(correlated, 1), size(correlated, 2)],
        "decorrelation_stride" => stride,
    )
end

function run_train_score(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :train_score, force)
        return Dict("status" => "skipped", "score_model_bson" => paths.score_model_bson)
    end

    require_artifact(paths.dataset_h5, :generate_data, :train_score)
    data = load_dataset(paths.dataset_h5)
    samples = data.uncorrelated
    normalized, μ, σ = normalize_dataset(samples)

    training = config["training"]
    gpu_ids = available_gpu_ids()
    use_gpu = parse_bool(get(training, "use_gpu", true)) && !isempty(gpu_ids)
    gpu_id = Int(get(training, "gpu_id", isempty(gpu_ids) ? 0 : first(gpu_ids)))
    if use_gpu
        set_gpu!(gpu_id)
    end

    noise_scales = Float64.(training["noise_scales"])
    epochs_schedule = Int.(training["epochs_schedule"])
    length(noise_scales) == length(epochs_schedule) || error("training.noise_scales and training.epochs_schedule must have the same length.")

    nn = nothing
    opt_state = nothing
    losses = Float32[]
    stage_offsets = Int[]
    t0 = time()

    for stage_idx in eachindex(noise_scales)
        push!(stage_offsets, length(losses))
        stage_seed = Int(training["seed"]) + 10_000 * (stage_idx - 1)
        Random.seed!(stage_seed)
        nn, stage_losses, _, _, _, _, opt_state = ScoreEstimation.train(
            normalized;
            preprocessing=false,
            σ=noise_scales[stage_idx],
            neurons=Int.(training["neurons"]),
            n_epochs=epochs_schedule[stage_idx],
            batch_size=Int(training["batch_size"]),
            lr=Float64(training["learning_rate"]),
            use_gpu=use_gpu,
            verbose=parse_bool(get(training, "verbose", true)),
            moment_weight_mean=Float64(get(training, "mean_regularization", 0.0)),
            moment_weight_stein=Float64(get(training, "stein_regularization", 0.0)),
            max_batches_per_epoch=Int(get(training, "max_batches_per_epoch", 0)),
            nn=nn,
            opt_state=opt_state,
        )
        append!(losses, Float32.(stage_losses))
    end

    elapsed = time() - t0
    metadata = Dict(
        "method" => "raw",
        "noise_scale" => noise_scales[end],
        "mean" => Vector{Float32}(μ),
        "scale" => Vector{Float32}(σ),
        "losses" => Float32.(losses),
        "training_time_seconds" => elapsed,
        "used_gpu" => use_gpu,
        "gpu_id" => gpu_id,
        "stage_noise_scales" => noise_scales,
        "stage_epochs" => epochs_schedule,
        "stage_loss_offsets" => stage_offsets,
        "neurons" => Int.(training["neurons"]),
        "batch_size" => Int(training["batch_size"]),
        "learning_rate" => Float64(training["learning_rate"]),
        "max_batches_per_epoch" => Int(get(training, "max_batches_per_epoch", 0)),
        "dataset_path" => paths.dataset_h5,
    )
    save_model(nn, paths.score_model_bson; metadata=metadata)

    h5open(paths.training_h5, "w") do file
        file["losses"] = metadata["losses"]
        file["training_time_seconds"] = elapsed
        file["noise_scale"] = metadata["noise_scale"]
        file["used_gpu"] = Int(use_gpu)
        file["gpu_id"] = gpu_id
        file["stage_noise_scales"] = metadata["stage_noise_scales"]
        file["stage_epochs"] = metadata["stage_epochs"]
        file["stage_loss_offsets"] = metadata["stage_loss_offsets"]
        file["mean"] = metadata["mean"]
        file["scale"] = metadata["scale"]
    end

    return Dict(
        "status" => "completed",
        "score_model_bson" => paths.score_model_bson,
        "training_h5" => paths.training_h5,
        "training_time_seconds" => elapsed,
        "used_gpu" => use_gpu,
        "gpu_id" => gpu_id,
        "final_loss" => Float64(losses[end]),
    )
end

function run_estimate_phi_sigma(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :estimate_phi_sigma, force)
        return Dict("status" => "skipped", "phi_sigma_h5" => paths.phi_sigma_h5)
    end

    require_artifact(paths.dataset_h5, :generate_data, :estimate_phi_sigma)
    require_artifact(paths.score_model_bson, :train_score, :estimate_phi_sigma)
    data = load_dataset(paths.dataset_h5)
    wrapper, metadata = build_score_wrapper(paths.score_model_bson)

    estimator_cfg = config["phi_sigma"]
    use_gpu = parse_bool(get(estimator_cfg, "use_gpu", true)) && CUDA.functional()
    if use_gpu
        gpu_id = Int(get(config["training"], "gpu_id", 1))
        set_gpu!(gpu_id)
    end

    perturb_override = Float64(get(estimator_cfg, "perturb_scale_override", -1.0))
    perturb_scale = perturb_override > 0 ? perturb_override : nothing
    Phi, Sigma, info = estimate_phi_sigma(
        data.correlated,
        wrapper;
        dt=data.sample_dt,
        stationary_samples=data.uncorrelated,
        perturb_scale=perturb_scale,
        generator_perturb_scale=Float64(get(estimator_cfg, "generator_perturb_scale", 0.0)),
        q_min_prob=Float64(estimator_cfg["q_min_prob"]),
        regularization=Float64(estimator_cfg["regularization"]),
        batch_size=Int(estimator_cfg["batch_size"]),
        use_gpu=use_gpu,
        use_stein_correction=parse_bool(get(estimator_cfg, "use_stein_correction", false)),
        enforce_psd_symmetric=parse_bool(get(estimator_cfg, "enforce_psd_symmetric", true)),
        finite_dt_correction=parse_bool(get(estimator_cfg, "finite_dt_correction", false)),
        finite_dt_correction_mode=Symbol(get(estimator_cfg, "finite_dt_correction_mode", "per_state")),
        rng=MersenneTwister(Int(estimator_cfg["seed"])),
    )

    audit_rng = MersenneTwister(Int(estimator_cfg["seed"]) + 101)
    audit_noise_scale = perturb_override > 0 ? perturb_override : Float64(metadata["noise_scale"])
    stationary_audit_samples = Float32.(data.uncorrelated .+ audit_noise_scale .* randn(audit_rng, Float32, size(data.uncorrelated)))
    stationary_score_values = batched_model_outputs(
        wrapper,
        stationary_audit_samples;
        batch_size=Int(estimator_cfg["batch_size"]),
        use_gpu=use_gpu,
    )
    V_audit = compute_score_position_matrix(
        stationary_audit_samples,
        wrapper;
        batch_size=Int(estimator_cfg["batch_size"]),
        use_gpu=use_gpu,
    )
    phi_from_neg_cdot = -Float64.(info[:cdot])
    phi_from_stein = Float64.(info[:cdot] / info[:V])
    phi_neg_cdot_gap = norm(Float64.(info[:Phi_raw]) .- phi_from_neg_cdot)
    phi_stein_gap = norm(Float64.(info[:Phi_raw]) .- phi_from_stein)
    stein_identity_error = norm(V_audit + Matrix{Float64}(I, size(V_audit, 1), size(V_audit, 2)))
    score_stats = norm_summary(stationary_score_values)

    h5open(paths.phi_sigma_h5, "w") do file
        file["Phi"] = Float64.(Phi)
        file["Sigma"] = Float64.(Sigma)
        file["Phi_raw"] = Float64.(info[:Phi_raw])
        file["Phi_symmetric"] = Float64.(info[:Phi_symmetric])
        file["Phi_symmetric_raw"] = Float64.(info[:Phi_symmetric_raw])
        file["Phi_antisymmetric"] = Float64.(info[:Phi_antisymmetric])
        file["cdot"] = Float64.(info[:cdot])
        file["V"] = Float64.(info[:V])
        file["centers"] = Float64.(info[:centers])
        file["pi_vec"] = Float64.(info[:pi_vec])
        file["counts"] = Int.(info[:counts])
        file["n_states"] = Int(info[:n_states])
        file["regularization_shift"] = Float64(info[:regularization_shift])
        file["noise_scale"] = Float64(metadata["noise_scale"])
        file["V_audit"] = Float64.(V_audit)
        file["phi_from_neg_cdot"] = phi_from_neg_cdot
        file["phi_from_stein"] = phi_from_stein
        file["phi_neg_cdot_gap"] = phi_neg_cdot_gap
        file["phi_stein_gap"] = phi_stein_gap
        file["stein_identity_error"] = stein_identity_error
        file["stationary_score_norm_mean"] = score_stats.mean
        file["stationary_score_norm_std"] = score_stats.std
        file["stationary_score_norm_q95"] = score_stats.q95
        file["stationary_score_norm_q99"] = score_stats.q99
        file["stationary_score_norm_max"] = score_stats.max
        write_sparse_matrix!(file, "Q", info[:Q])
    end

    return Dict(
        "status" => "completed",
        "phi_sigma_h5" => paths.phi_sigma_h5,
        "n_states" => Int(info[:n_states]),
        "regularization_shift" => Float64(info[:regularization_shift]),
        "stein_identity_error" => stein_identity_error,
        "phi_neg_cdot_gap" => phi_neg_cdot_gap,
        "phi_stein_gap" => phi_stein_gap,
    )
end

function run_observed_residuals(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :observed_residuals, force)
        return Dict("status" => "skipped", "observed_residuals_h5" => paths.observed_residuals_h5)
    end

    require_artifact(paths.dataset_h5, :generate_data, :observed_residuals)
    require_artifact(paths.score_model_bson, :train_score, :observed_residuals)
    require_artifact(paths.phi_sigma_h5, :estimate_phi_sigma, :observed_residuals)

    data = load_dataset(paths.dataset_h5)
    wrapper, _ = build_score_wrapper(paths.score_model_bson)
    residual_cfg = config["observed_residuals"]
    use_gpu = parse_bool(get(residual_cfg, "use_gpu", true)) && CUDA.functional()
    if use_gpu
        gpu_id = Int(get(config["training"], "gpu_id", 1))
        set_gpu!(gpu_id)
    end

    tD_override = get(residual_cfg, "tD_override", nothing)
    lag_max_step_base = resolve_lag_max_step(data, tD_override)
    lag_max_step = clamp(
        round(Int, Float64(residual_cfg["max_lag_fraction"]) * lag_max_step_base),
        1,
        lag_max_step_base,
    )
    lag_start = parse_bool(get(residual_cfg, "include_zero_lag", true)) ? 0 : 1
    lag_step_stride = max(Int(residual_cfg["lag_step_stride"]), 1)
    lag_steps = collect(lag_start:lag_step_stride:lag_max_step)
    last(lag_steps) == lag_max_step || push!(lag_steps, lag_max_step)
    lag_times = Float64.(lag_steps) .* data.sample_dt
    smooth_window_radius = Int(get(residual_cfg, "empirical_smoothing_window_radius", min(3, max(length(lag_steps) ÷ 8, 1))))
    smooth_degree = Int(get(residual_cfg, "empirical_smoothing_degree", 2))

    score_values = batched_model_outputs(
        wrapper,
        data.correlated;
        batch_size=Int(residual_cfg["batch_size"]),
        use_gpu=use_gpu,
    )

    phi_sigma = h5open(paths.phi_sigma_h5, "r") do file
        (
            Phi=Float64.(read(file["Phi"])),
            Q=read_sparse_matrix(file, "Q"),
            centers=Float64.(read(file["centers"])),
            pi_vec=Float64.(read(file["pi_vec"])),
            n_states=Int(read(file["n_states"])),
        )
    end

    lag_steps_sorted, generator_series = generator_state_derivative_series(
        phi_sigma.Q,
        phi_sigma.centers,
        phi_sigma.pi_vec,
        lag_steps,
        data.sample_dt,
    )
    generator_lookup = Dict(lag => idx for (idx, lag) in enumerate(lag_steps_sorted))
    observable_cfg = get(config, "delta_m_observables", Dict{String,Any}())
    observables = build_delta_m_observable_specs(data.uncorrelated, observable_cfg)
    observable_results = NamedTuple[]

    h5open(paths.observed_residuals_h5, "w") do file
        file["lag_steps"] = Int.(lag_steps)
        file["lag_times"] = Float64.(lag_times)
        file["Phi"] = Float64.(phi_sigma.Phi)
        file["n_states"] = phi_sigma.n_states
        file["score_eval_use_gpu"] = Int(use_gpu)
        file["cdot_mode"] = String(get(residual_cfg, "cdot_mode", "dual"))
        file["empirical_smoothing_window_radius"] = smooth_window_radius
        file["empirical_smoothing_degree"] = smooth_degree

        for obs in observables
            Xi = Float64.(evaluate_observable(obs, phi_sigma.centers))
            Sphi = empirical_score_observable_tensor(obs, data.correlated, score_values, lag_steps)
            Cphi = empirical_observable_correlation_tensor(obs, data.correlated, lag_steps)
            Cphi_smoothed, Cdot_emp = smooth_tensor_time_derivative(
                Cphi,
                lag_times;
                window_radius=smooth_window_radius,
                degree=smooth_degree,
            )
            output_dim = length(obs.output_labels)
            dim = size(phi_sigma.Phi, 1)
            Cdot_q = Array{Float64}(undef, output_dim, dim, length(lag_steps))
            SphiPhi = Array{Float64}(undef, output_dim, dim, length(lag_steps))
            Ephi_q = Array{Float64}(undef, output_dim, dim, length(lag_steps))
            Ephi_emp = Array{Float64}(undef, output_dim, dim, length(lag_steps))

            for (lag_idx, lag_step) in enumerate(lag_steps)
                generator_idx = generator_lookup[Int(lag_step)]
                @views Cdot_q[:, :, lag_idx] .= Xi * generator_series[:, :, generator_idx]
                @views SphiPhi[:, :, lag_idx] .= Sphi[:, :, lag_idx] * phi_sigma.Phi
                @views Ephi_q[:, :, lag_idx] .= SphiPhi[:, :, lag_idx] .- Cdot_q[:, :, lag_idx]
                @views Ephi_emp[:, :, lag_idx] .= SphiPhi[:, :, lag_idx] .- Cdot_emp[:, :, lag_idx]
            end

            fro_norm_q = [norm(@view Ephi_q[:, :, lag_idx]) for lag_idx in eachindex(lag_steps)]
            fro_norm_emp = [norm(@view Ephi_emp[:, :, lag_idx]) for lag_idx in eachindex(lag_steps)]
            cdot_rel_error = relative_frobenius_error_by_slice(Cdot_q, Cdot_emp)
            cdot_diff_norm = [norm(Float64.(@view Cdot_q[:, :, lag_idx]) .- Float64.(@view Cdot_emp[:, :, lag_idx])) for lag_idx in eachindex(lag_steps)]
            group = create_group(file, slugify(obs.name))
            group["name"] = obs.name
            group["Xi"] = Xi
            group["Sphi"] = Sphi
            group["Cphi"] = Cphi
            group["Cphi_smoothed"] = Cphi_smoothed
            group["SphiPhi"] = SphiPhi
            group["Cdot"] = Cdot_q
            group["Cdot_q"] = Cdot_q
            group["Cdot_emp"] = Cdot_emp
            group["Ephi"] = Ephi_q
            group["Ephi_q"] = Ephi_q
            group["Ephi_emp"] = Ephi_emp
            group["fro_norm"] = Float64.(fro_norm_q)
            group["fro_norm_q"] = Float64.(fro_norm_q)
            group["fro_norm_emp"] = Float64.(fro_norm_emp)
            group["cdot_relative_error"] = Float64.(cdot_rel_error)
            group["cdot_difference_norm"] = Float64.(cdot_diff_norm)
            group["output_dim"] = output_dim
            for (label_idx, label) in enumerate(obs.output_labels)
                group["output_label_$(label_idx)"] = label
            end

            push!(observable_results, (
                name=obs.name,
                output_labels=obs.output_labels,
                Xi=Xi,
                Sphi=Sphi,
                Cphi=Cphi,
                Cphi_smoothed=Cphi_smoothed,
                SphiPhi=SphiPhi,
                Cdot=Cdot_q,
                Cdot_q=Cdot_q,
                Cdot_emp=Cdot_emp,
                Ephi=Ephi_q,
                Ephi_q=Ephi_q,
                Ephi_emp=Ephi_emp,
                fro_norm=Float64.(fro_norm_q),
                fro_norm_q=Float64.(fro_norm_q),
                fro_norm_emp=Float64.(fro_norm_emp),
                cdot_relative_error=Float64.(cdot_rel_error),
                cdot_difference_norm=Float64.(cdot_diff_norm),
                max_abs=maximum(abs, Ephi_q),
            ))
        end
    end

    fig = observable_residual_figure(observable_results, lag_times)
    save(paths.observed_residuals_png, fig, px_per_unit=2)
    save(paths.observed_residuals_pdf, fig)

    coordinate_result = first(observable_results)
    zero_lag_idx = findfirst(==(0), lag_steps)
    coordinate_zero_lag_norm = zero_lag_idx === nothing ? NaN : coordinate_result.fro_norm[zero_lag_idx]

    return Dict(
        "status" => "completed",
        "observed_residuals_h5" => paths.observed_residuals_h5,
        "observed_residuals_png" => paths.observed_residuals_png,
        "observed_residuals_pdf" => paths.observed_residuals_pdf,
        "n_lags" => length(lag_steps),
        "observables" => [result.name for result in observable_results],
        "coordinate_zero_lag_norm" => coordinate_zero_lag_norm,
        "max_residual_abs" => maximum(result.max_abs for result in observable_results),
        "max_cdot_relative_error" => maximum(maximum(result.cdot_relative_error) for result in observable_results),
    )
end

function run_overview(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :overview, force)
        return Dict("status" => "skipped", "overview_png" => paths.overview_png)
    end

    require_artifact(paths.dataset_h5, :generate_data, :overview)
    require_artifact(paths.score_model_bson, :train_score, :overview)
    require_artifact(paths.phi_sigma_h5, :estimate_phi_sigma, :overview)
    data = load_dataset(paths.dataset_h5)
    eval_cfg = build_eval_config(data.cfg, config)
    wrapper, metadata = build_score_wrapper(paths.score_model_bson)

    phi_sigma = h5open(paths.phi_sigma_h5, "r") do file
        (
            Phi=Float32.(read(file["Phi"])),
            Sigma=Float32.(read(file["Sigma"])),
            V=Float64.(read(file["V"])),
            Phi_symmetric=Float64.(read(file["Phi_symmetric"])),
            Phi_antisymmetric=Float64.(read(file["Phi_antisymmetric"])),
            n_states=Int(read(file["n_states"])),
            regularization_shift=Float64(read(file["regularization_shift"])),
        )
    end

    integration = config["integration"]
    device = parse_bool(get(integration, "use_gpu", true)) && CUDA.functional() ? :gpu : :cpu
    if device === :gpu
        gpu_id = Int(get(config["training"], "gpu_id", 1))
        set_gpu!(gpu_id)
    end

    x0 = sample_initial_conditions(data.uncorrelated, Int(integration["n_ensembles"]), Int(integration["seed"]))
    snapshots = evolve_affine_score_langevin_snapshots(
        wrapper,
        x0,
        phi_sigma.Phi,
        phi_sigma.Sigma;
        dt=Float64(integration["dt"]),
        n_steps=Int(integration["n_steps"]),
        burn_in=Int(integration["burn_in"]),
        resolution=Int(integration["resolution"]),
        device=device,
        seed=Int(integration["seed"]),
        progress=parse_bool(get(integration, "progress", false)),
        progress_desc="Raw DSM reduced Langevin",
    )

    stationary = stationary_from_snapshots(snapshots)
    trajectory = first_trajectory(snapshots)
    true_diag = build_brusselator_diagnostics(eval_cfg, data.correlated, data.uncorrelated; sample_dt=data.sample_dt)
    sample_dt = Float64(integration["dt"]) * Int(integration["resolution"])
    model_diag = build_brusselator_diagnostics(
        eval_cfg,
        trajectory,
        stationary;
        sample_dt=sample_dt,
        x_range=true_diag.x_range,
        y_range=true_diag.y_range,
    )

    diagnostics = config["diagnostics"]
    acf_window = findlast(<=(Float64(diagnostics["metric_time_horizon"])), true_diag.lag_axis)
    acf_window = something(acf_window, length(true_diag.lag_axis))
    ccf_window = count(abs.(true_diag.ccf_axis) .<= Float64(diagnostics["metric_time_horizon"]))
    metrics = compare_brusselator_diagnostics(true_diag, model_diag; acf_window=acf_window, ccf_window=ccf_window)

    fig = single_method_figure(model_diag, trajectory, metadata, metrics, phi_sigma, true_diag, data.correlated, data.time_axis)
    save(paths.overview_png, fig, px_per_unit=2)
    save(paths.overview_pdf, fig)

    h5open(paths.diagnostics_h5, "w") do file
        file["stationary_samples"] = stationary
        file["trajectory"] = trajectory
        file["Phi"] = Float64.(phi_sigma.Phi)
        file["Sigma"] = Float64.(phi_sigma.Sigma)
        file["kl_x"] = metrics.kl_x
        file["kl_y"] = metrics.kl_y
        file["kl_xy"] = metrics.kl_xy
        file["rmse_acf_x"] = metrics.rmse_acf_x
        file["rmse_acf_y"] = metrics.rmse_acf_y
        file["rmse_ccf"] = metrics.rmse_ccf
        file["training_time_seconds"] = metadata["training_time_seconds"]
    end

    update_results_summary!(paths, Dict(
        "kl_x" => metrics.kl_x,
        "kl_y" => metrics.kl_y,
        "kl_xy" => metrics.kl_xy,
        "rmse_acf_x" => metrics.rmse_acf_x,
        "rmse_acf_y" => metrics.rmse_acf_y,
        "rmse_ccf" => metrics.rmse_ccf,
    ))

    return Dict(
        "status" => "completed",
        "overview_png" => paths.overview_png,
        "overview_pdf" => paths.overview_pdf,
        "diagnostics_h5" => paths.diagnostics_h5,
        "kl_xy" => metrics.kl_xy,
        "rmse_acf_x" => metrics.rmse_acf_x,
        "rmse_acf_y" => metrics.rmse_acf_y,
        "rmse_ccf" => metrics.rmse_ccf,
    )
end

function run_joint_score(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :joint_score, force)
        return Dict("status" => "skipped", "joint_score_png" => paths.joint_score_png)
    end

    require_artifact(paths.dataset_h5, :generate_data, :joint_score)
    require_artifact(paths.score_model_bson, :train_score, :joint_score)
    data = load_dataset(paths.dataset_h5)
    joint_cfg = config["joint_score"]
    tD_override = get(joint_cfg, "tD_override", nothing)
    lag_max_step = resolve_lag_max_step(data, tD_override)
    lag_steps = collect(1:lag_max_step)
    lag_times = Float64.(lag_steps) .* data.sample_dt
    tD = lag_times[end]
    lag_sampling_power = Float64(get(joint_cfg, "lag_sampling_power", 0.0))
    lag_cdf = if lag_sampling_power == 0
        nothing
    else
        raw_weights = Float64.(lag_steps) .^ (-lag_sampling_power)
        weights = raw_weights ./ sum(raw_weights)
        cumsum(weights)
    end

    μ4, σ4 = pair_normalization_stats(data.uncorrelated)
    gpu_ids = available_gpu_ids()
    use_gpu = parse_bool(get(joint_cfg, "use_gpu", true)) && !isempty(gpu_ids)
    gpu_id = Int(get(joint_cfg, "gpu_id", isempty(gpu_ids) ? 0 : first(gpu_ids)))
    if use_gpu
        set_gpu!(gpu_id)
    end

    noise_scales = Float64.(joint_cfg["noise_scales"])
    epochs_schedule = Int.(joint_cfg["epochs_schedule"])
    learning_rates = haskey(joint_cfg, "learning_rates") ? Float64.(joint_cfg["learning_rates"]) : fill(Float64(joint_cfg["learning_rate"]), length(noise_scales))
    length(noise_scales) == length(epochs_schedule) || error("joint_score.noise_scales and joint_score.epochs_schedule must have the same length.")
    length(noise_scales) == length(learning_rates) || error("joint_score.learning_rates must match joint_score.noise_scales when provided.")
    nn = nothing
    opt_state = nothing
    training_losses = Float32[]
    stage_offsets = Int[]
    stage_noise_scales = Float64[]
    stage_epochs = Int[]
    stage_learning_rates = Float64[]
    rng = MersenneTwister(Int(joint_cfg["seed"]))
    use_gpu_eff = use_gpu && CUDA.functional()
    mean_reg = Float64(get(joint_cfg, "mean_regularization", 0.0))
    stein_reg = Float64(get(joint_cfg, "stein_regularization", 0.0))
    reset_optimizer_each_stage = parse_bool(get(joint_cfg, "reset_optimizer_each_stage", false))
    time_frequencies = haskey(joint_cfg, "time_frequencies") ? Float32.(joint_cfg["time_frequencies"]) : Float32[1, 2, 4, 8]
    t0 = time()

    for stage_idx in eachindex(noise_scales)
        noise_scale = noise_scales[stage_idx]
        n_epochs = epochs_schedule[stage_idx]
        learning_rate = learning_rates[stage_idx]
        push!(stage_offsets, length(training_losses))
        push!(stage_noise_scales, noise_scale)
        push!(stage_epochs, n_epochs)
        push!(stage_learning_rates, learning_rate)

        if nn === nothing
            nn = JointResidualScoreModel(
                nothing,
                μ4,
                σ4,
                noise_scale,
                Int.(joint_cfg["neurons"]);
                time_frequencies=time_frequencies,
            )
        else
            nn = JointResidualScoreModel(
                nn.correction_model,
                nothing,
                μ4,
                σ4,
                noise_scale,
                time_frequencies,
            )
        end
        nn = Flux.f32(nn)
        nn = use_gpu_eff ? ScoreEstimation._move_to_gpu(nn, CUDA) : nn
        if opt_state === nothing || reset_optimizer_each_stage
            opt_state = Flux.setup(Flux.Adam(Float32(learning_rate)), nn)
        end

        verbose = parse_bool(get(joint_cfg, "verbose", true))
        verbose && println("[joint_score] stage $(stage_idx)/$(length(noise_scales)) -> σ=$(noise_scale), epochs=$(n_epochs), lr=$(learning_rate)")
        verbose && (progress = ScoreEstimation.ProgressMeter.Progress(n_epochs; desc="Joint-score epochs", showspeed=false))

        for epoch in 1:n_epochs
            epoch_losses = Float32[]
            for _ in 1:Int(joint_cfg["max_batches_per_epoch"])
                y_clean, tau = build_joint_training_batch(
                    data.correlated,
                    lag_steps,
                    lag_cdf,
                    μ4,
                    σ4,
                    Int(joint_cfg["batch_size"]),
                    rng,
                )
                ϵ = randn(rng, Float32, size(y_clean))
                y_noisy = @. y_clean + Float32(noise_scale) * ϵ
                input = vcat(y_noisy, tau)
                input_dev = ScoreEstimation._to_device(input; use_gpu=use_gpu_eff)
                ϵ_dev = ScoreEstimation._to_device(ϵ; use_gpu=use_gpu_eff)
                y_noisy_dev = view(input_dev, 1:4, :)
                loss, grads = Flux.withgradient(nn) do model
                    pred = model(input_dev)
                    Flux.mse(pred, ϵ_dev) + ScoreEstimation._score_moment_penalty(
                        pred,
                        y_noisy_dev,
                        noise_scale;
                        mean_weight=mean_reg,
                        stein_weight=stein_reg,
                    )
                end
                Flux.update!(opt_state, nn, grads[1])
                push!(epoch_losses, Float32(loss))
            end
            push!(training_losses, mean(epoch_losses))
            verbose && ScoreEstimation.ProgressMeter.next!(progress)
        end
        verbose && ScoreEstimation.ProgressMeter.finish!(progress)
    end

    elapsed = time() - t0
    final_noise_scale = noise_scales[end]
    metadata = Dict(
        "method" => "joint_score",
        "noise_scale" => final_noise_scale,
        "mean" => Vector{Float32}(μ4),
        "scale" => Vector{Float32}(σ4),
        "losses" => Float32.(training_losses),
        "training_time_seconds" => elapsed,
        "used_gpu" => use_gpu,
        "gpu_id" => gpu_id,
        "stage_noise_scales" => stage_noise_scales,
        "stage_epochs" => stage_epochs,
        "stage_learning_rates" => stage_learning_rates,
        "stage_loss_offsets" => stage_offsets,
        "lag_steps" => Int.(lag_steps),
        "lag_times" => Float64.(lag_times),
        "tD" => tD,
        "time_frequencies" => Vector{Float32}(time_frequencies),
        "neurons" => Int.(joint_cfg["neurons"]),
        "batch_size" => Int(joint_cfg["batch_size"]),
        "learning_rate" => Float64(last(learning_rates)),
        "max_batches_per_epoch" => Int(joint_cfg["max_batches_per_epoch"]),
        "dataset_path" => paths.dataset_h5,
    )
    save_model(Flux.cpu(nn), paths.joint_score_model_bson; metadata=metadata)

    loss_by_lag = estimate_joint_dsm_loss(
        Flux.cpu(nn),
        data.correlated,
        lag_steps,
        μ4,
        σ4,
        final_noise_scale;
        batch_size=Int(joint_cfg["batch_size"]),
        eval_batches_per_lag=Int(joint_cfg["eval_batches_per_lag"]),
        use_gpu=false,
        seed=Int(joint_cfg["seed"]) + 99,
    )

    selected_lag_steps = representative_lag_steps(lag_max_step, joint_cfg["lag_fractions"])
    contour_levels = Int(joint_cfg["contour_levels"])
    stationary_wrapper, _ = build_score_wrapper(paths.score_model_bson)
    conditional_wrapper = ConditionalScoreWrapper(
        Flux.cpu(nn),
        stationary_wrapper,
        Float32.(μ4),
        Float32.(σ4),
        Float32(final_noise_scale),
        Float32(tD),
    )
    conditional_audit_pairs = min(Int(get(joint_cfg, "conditional_audit_pairs", 20_000)), size(data.correlated, 2) - 1)
    sampler_device = parse_bool(get(joint_cfg, "sampler_use_gpu", true)) && CUDA.functional() ? :gpu : :cpu
    if sampler_device === :gpu
        set_gpu!(gpu_id)
    end

    rows = Any[]
    h5open(paths.joint_score_h5, "w") do file
        file["training_losses"] = Float32.(training_losses)
        file["stage_noise_scales"] = Float64.(stage_noise_scales)
        file["stage_epochs"] = Int.(stage_epochs)
        file["stage_learning_rates"] = Float64.(stage_learning_rates)
        file["stage_loss_offsets"] = Int.(stage_offsets)
        file["lag_steps"] = Int.(lag_steps)
        file["lag_times"] = Float64.(lag_times)
        file["loss_by_lag"] = Float64.(loss_by_lag)
        file["tD"] = tD
        file["mean"] = Vector{Float32}(μ4)
        file["scale"] = Vector{Float32}(σ4)

        for lag_step in selected_lag_steps
            lag_time = lag_step * data.sample_dt
            tau = Float32(lag_step / lag_max_step)
            obs_pairs_raw = pair_samples_for_lag(data.correlated, lag_step)
            obs_pairs_norm = normalize_pair_samples(obs_pairs_raw, μ4, σ4)
            x0 = sample_initial_conditions(obs_pairs_norm, Int(joint_cfg["sampler_ensembles"]), Int(joint_cfg["seed"]) + lag_step)
            wrapper = JointPairScoreSliceModel(Flux.cpu(nn), final_noise_scale, tau)
            snapshots = evolve_score_langevin_snapshots(
                wrapper,
                x0;
                dt=Float64(joint_cfg["sampler_dt"]),
                n_steps=Int(joint_cfg["sampler_steps"]),
                burn_in=Int(joint_cfg["sampler_burn_in"]),
                resolution=Int(joint_cfg["sampler_resolution"]),
                device=sampler_device,
                seed=Int(joint_cfg["seed"]) + 1000 + lag_step,
                progress=parse_bool(get(joint_cfg, "sampler_progress", false)),
                progress_desc="Joint score sampler (lag=$(lag_step))",
                boundary=(-Float64(joint_cfg["sampler_boundary"]), Float64(joint_cfg["sampler_boundary"])),
            )
            model_pairs_norm = stationary_from_snapshots(snapshots)
            model_pairs_raw = denormalize_pair_samples(model_pairs_norm, μ4, σ4)
            V = stein_matrix(obs_pairs_norm, wrapper; batch_size=Int(joint_cfg["batch_size"]), use_gpu=false)
            stein_error = norm(V + Matrix{Float64}(I, 4, 4))

            lag_group = create_group(file, "lag_$(lpad(string(lag_step), 4, '0'))")
            lag_group["lag_step"] = lag_step
            lag_group["lag_time"] = lag_time
            lag_group["stein_matrix"] = V
            lag_group["stein_error"] = stein_error
            lag_group["first_block_stein_error"] = norm(@view(V[1:2, 1:2]) + Matrix{Float64}(I, 2, 2))

            x0_audit, xt_audit = lagged_pairs_matrix(data.correlated, lag_step; max_pairs=conditional_audit_pairs, rng=rng)
            helper_cond = conditional_score(conditional_wrapper, x0_audit, xt_audit, lag_time)
            μ = reshape(conditional_wrapper.pair_mean, :, 1)
            σ = reshape(conditional_wrapper.pair_scale, :, 1)
            y0 = (x0_audit .- @view(μ[1:2, :])) ./ @view(σ[1:2, :])
            yt = (xt_audit .- @view(μ[3:4, :])) ./ @view(σ[3:4, :])
            tau_norm = fill(Float32(lag_time / tD), 1, size(x0_audit, 2))
            eps_hat = conditional_wrapper.joint_model(vcat(y0, yt, tau_norm))
            joint_score = (-eps_hat ./ Float32(final_noise_scale))
            manual_cond = @view(joint_score[1:2, :]) ./ @view(σ[1:2, :]) .- stationary_wrapper(x0_audit)
            conditional_stats = norm_summary(helper_cond)
            lag_group["conditional_score_norm_mean"] = conditional_stats.mean
            lag_group["conditional_score_norm_std"] = conditional_stats.std
            lag_group["conditional_score_norm_q95"] = conditional_stats.q95
            lag_group["conditional_score_norm_q99"] = conditional_stats.q99
            lag_group["conditional_score_norm_max"] = conditional_stats.max
            lag_group["conditional_manual_consistency_error"] = maximum(abs.(Float64.(helper_cond .- manual_cond)))

            projections = NamedTuple[]
            for (proj_idx, (i, j, label)) in enumerate(JOINT_PAIR_PROJECTIONS)
                xs_obs = vec(obs_pairs_raw[i, :])
                ys_obs = vec(obs_pairs_raw[j, :])
                x_range = vector_value_range(xs_obs)
                y_range = vector_value_range(ys_obs)
                obs_pdf = estimate_bivariate_pdf_histogram(xs_obs, ys_obs; nbins=Int(joint_cfg["joint_pdf_bins"]), x_range=x_range, y_range=y_range)
                model_pdf = estimate_bivariate_pdf_histogram(vec(model_pairs_raw[i, :]), vec(model_pairs_raw[j, :]); nbins=Int(joint_cfg["joint_pdf_bins"]), x_range=x_range, y_range=y_range)
                metrics = pair_pdf_metrics(obs_pdf, model_pdf)
                proj_group = create_group(lag_group, "projection_$(proj_idx)")
                proj_group["label"] = label
                proj_group["kl"] = metrics.kl
                proj_group["l1"] = metrics.l1
                proj_group["obs_density"] = obs_pdf.density
                proj_group["model_density"] = model_pdf.density
                proj_group["x"] = obs_pdf.x
                proj_group["y"] = obs_pdf.y
                push!(projections, (
                    label=label,
                    xlabel=label[1:4],
                    ylabel=label[end-3:end],
                    obs_pdf=obs_pdf,
                    model_pdf=model_pdf,
                    metrics=metrics,
                ))
            end

            push!(rows, (
                lag_step=lag_step,
                lag_time=lag_time,
                stein_matrix=V,
                stein_error=stein_error,
                first_block_stein_error=norm(@view(V[1:2, 1:2]) + Matrix{Float64}(I, 2, 2)),
                conditional_score_stats=conditional_stats,
                conditional_manual_consistency_error=maximum(abs.(Float64.(helper_cond .- manual_cond))),
                contour_levels=contour_levels,
                projections=projections,
            ))
        end
    end

    fig = joint_score_figure(rows, training_losses, stage_offsets, stage_noise_scales, lag_steps, lag_times, loss_by_lag, tD)
    save(paths.joint_score_png, fig, px_per_unit=2)
    save(paths.joint_score_pdf, fig)

    mean_projection_kl = mean([proj.metrics.kl for row in rows for proj in row.projections])
    max_stein_error = maximum(getfield.(rows, :stein_error))
    max_first_block_stein_error = maximum(getfield.(rows, :first_block_stein_error))
    max_conditional_manual_consistency_error = maximum(getfield.(rows, :conditional_manual_consistency_error))
    mean_conditional_score_norm = mean([row.conditional_score_stats.mean for row in rows])

    return Dict(
        "status" => "completed",
        "joint_score_model_bson" => paths.joint_score_model_bson,
        "joint_score_h5" => paths.joint_score_h5,
        "joint_score_png" => paths.joint_score_png,
        "joint_score_pdf" => paths.joint_score_pdf,
        "tD" => tD,
        "mean_projection_kl" => mean_projection_kl,
        "max_stein_error" => max_stein_error,
        "max_first_block_stein_error" => max_first_block_stein_error,
        "max_conditional_manual_consistency_error" => max_conditional_manual_consistency_error,
        "mean_conditional_score_norm" => mean_conditional_score_norm,
        "training_time_seconds" => elapsed,
    )
end

function evaluate_delta_m_residuals(model,
                                    score_wrapper,
                                    conditional_wrapper,
                                    observables::AbstractVector{ObservableSpec},
                                    targets::Dict{String,Any},
                                    eval_pairs::Dict{Int,Any},
                                    lag_times::AbstractVector{<:Real},
                                    lag_steps::AbstractVector{<:Integer};
                                    use_gpu::Bool=false,
                                    seed::Integer=0)
    rng = MersenneTwister(seed)
    model_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(model), CUDA)
    else
        deepcopy(model)
    end
    score_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(score_wrapper), CUDA)
    else
        deepcopy(score_wrapper)
    end
    cond_eval = if use_gpu && CUDA.functional()
        ScoreEstimation._move_to_gpu(deepcopy(conditional_wrapper), CUDA)
    else
        deepcopy(conditional_wrapper)
    end
    Flux.testmode!(model_eval)
    Flux.testmode!(score_eval)
    Flux.testmode!(cond_eval)

    results = NamedTuple[]
    for obs in observables
        target = targets[obs.name]
        E_obs = Float64.(target.E_obs)
        E_model = Array{Float64}(undef, size(E_obs))
        rel_error = Vector{Float64}(undef, length(lag_steps))
        for (lag_idx, lag_step) in enumerate(lag_steps)
            x0_cpu, xt_cpu = eval_pairs[Int(lag_step)]
            lag_time = lag_times[lag_idx]
            x0 = ScoreEstimation._to_device(x0_cpu; use_gpu=use_gpu)
            xt = ScoreEstimation._to_device(xt_cpu; use_gpu=use_gpu)
            scond = conditional_score(cond_eval, x0, xt, lag_time)
            score0 = score_eval(x0)
            comps = mobility_components(model_eval, x0, score0)
            eff = effective_delta_score(comps, scond)
            phi_xt = evaluate_observable_device(obs, xt)
            E_model_batch = Array((phi_xt * transpose(eff)) ./ size(x0, 2))
            @views E_model[:, :, lag_idx] .= Float64.(E_model_batch)
            rel_error[lag_idx] = norm(E_model[:, :, lag_idx] .- E_obs[:, :, lag_idx]) / max(norm(E_obs[:, :, lag_idx]), eps(Float64))
        end
        push!(results, (
            name=obs.name,
            output_labels=obs.output_labels,
            group=obs.group,
            E_obs=E_obs,
            E_model=E_model,
            relative_error=rel_error,
            relative_rmse=relative_frobenius_rmse(E_obs, E_model),
        ))
    end

    return results
end

function summarize_delta_m_samples(model::DeltaMMobilityField,
                                   score_wrapper,
                                   samples::AbstractMatrix)
    comps = batched_mobility_components(model, score_wrapper, samples; batch_size=8192, use_gpu=false)
    mean_delta = Float64.([
        mean(comps[:delta11]) mean(comps[:delta12]);
        mean(comps[:delta21]) mean(comps[:delta22]);
    ])
    min_lambda = minimum(
        0.5f0 .* ((comps[:m11] .+ comps[:m22]) .-
        sqrt.(max.((comps[:m11] .- comps[:m22]) .^ 2 .+ 4f0 .* ((0.5f0 .* (comps[:m12] .+ comps[:m21])) .^ 2), 0f0)))
    )
    skew_error = maximum(abs.(comps[:m12] .- comps[:m21] .- 2f0 .* comps[:omega]))
    return (
        mean_delta=mean_delta,
        mean_delta_norm=norm(mean_delta),
        min_lambda=Float64(min_lambda),
        skew_error=Float64(skew_error),
    )
end

function prepare_delta_m_eval_pairs(correlated::AbstractMatrix,
                                    lag_steps::AbstractVector{<:Integer};
                                    max_pairs::Integer,
                                    seed::Integer)
    rng = MersenneTwister(seed)
    out = Dict{Int,Any}()
    for lag_step in lag_steps
        out[Int(lag_step)] = lagged_pairs_matrix(correlated, Int(lag_step); max_pairs=max_pairs, rng=rng)
    end
    return out
end

function delta_m_lag_buckets(lag_steps::AbstractVector{<:Integer},
                             lag_times::AbstractVector{<:Real})
    tmax = maximum(Float64.(lag_times))
    t1 = tmax / 3
    t2 = 2tmax / 3
    short = Int[]
    medium = Int[]
    long = Int[]
    for (step, t) in zip(Int.(lag_steps), Float64.(lag_times))
        if t <= t1
            push!(short, step)
        elseif t <= t2
            push!(medium, step)
        else
            push!(long, step)
        end
    end
    isempty(short) && append!(short, Int.(lag_steps[1:min(end, 1)]))
    isempty(medium) && append!(medium, Int.(lag_steps[clamp(length(lag_steps) ÷ 2, 1, length(lag_steps)):clamp(length(lag_steps) ÷ 2, 1, length(lag_steps))]))
    isempty(long) && append!(long, Int.(lag_steps[max(end, 1):end]))
    return (short=short, medium=medium, long=long)
end

function build_smoothness_context(samples::AbstractMatrix,
                                  score_wrapper,
                                  cfg::Dict{String,Any};
                                  batch_size::Integer=8192)
    axes = percentile_field_axes(
        samples,
        Int(get(cfg, "smoothness_grid_nx", 18)),
        Int(get(cfg, "smoothness_grid_ny", 18));
        lower_quantile=Float64(get(cfg, "smoothness_lower_quantile", 0.01)),
        upper_quantile=Float64(get(cfg, "smoothness_upper_quantile", 0.99)),
        padding_fraction=Float64(get(cfg, "smoothness_padding_fraction", 0.04)),
    )
    xgrid = axes.x
    ygrid = axes.y
    pairs_a = Vector{NTuple{2,Float32}}()
    pairs_b = Vector{NTuple{2,Float32}}()
    for iy in eachindex(ygrid), ix in 1:(length(xgrid) - 1)
        push!(pairs_a, (Float32(xgrid[ix]), Float32(ygrid[iy])))
        push!(pairs_b, (Float32(xgrid[ix + 1]), Float32(ygrid[iy])))
    end
    for iy in 1:(length(ygrid) - 1), ix in eachindex(xgrid)
        push!(pairs_a, (Float32(xgrid[ix]), Float32(ygrid[iy])))
        push!(pairs_b, (Float32(xgrid[ix]), Float32(ygrid[iy + 1])))
    end
    xa = hcat([[p[1], p[2]] for p in pairs_a]...)
    xb = hcat([[p[1], p[2]] for p in pairs_b]...)
    sa = Float32.(batched_model_outputs(score_wrapper, xa; batch_size=batch_size, use_gpu=false))
    sb = Float32.(batched_model_outputs(score_wrapper, xb; batch_size=batch_size, use_gpu=false))
    return (xa=Float32.(xa), xb=Float32.(xb), sa=sa, sb=sb)
end

function delta_m_mean_penalty(comps)
    mean_delta = Float32.([
        mean(comps.delta11) mean(comps.delta12);
        mean(comps.delta21) mean(comps.delta22);
    ])
    return sum(abs2, mean_delta)
end

function delta_m_anchor_penalty(comps)
    return mean(abs2, comps.delta11) + mean(abs2, comps.delta12) + mean(abs2, comps.delta21) + mean(abs2, comps.delta22)
end

function delta_m_psd_margin_penalty(comps, floor::Real)
    sym12 = 0.5f0 .* (comps.m12 .+ comps.m21)
    λmin = 0.5f0 .* ((comps.m11 .+ comps.m22) .- sqrt.(max.((comps.m11 .- comps.m22) .^ 2 .+ 4f0 .* (sym12 .^ 2), 0f0)))
    return mean(abs2, max.(Float32(floor) .- λmin, 0f0))
end

function delta_m_smoothness_penalty(model,
                                    xa,
                                    sa,
                                    xb,
                                    sb)
    comps_a = mobility_components(model, xa, sa)
    comps_b = mobility_components(model, xb, sb)
    return mean(abs2, comps_a.delta11 .- comps_b.delta11) +
        mean(abs2, comps_a.delta12 .- comps_b.delta12) +
        mean(abs2, comps_a.delta21 .- comps_b.delta21) +
        mean(abs2, comps_a.delta22 .- comps_b.delta22)
end

function build_divergence_context(samples::AbstractMatrix,
                                  score_wrapper,
                                  cfg::Dict{String,Any};
                                  batch_size::Integer=8192)
    axes = percentile_field_axes(
        samples,
        Int(get(cfg, "smoothness_grid_nx", 18)),
        Int(get(cfg, "smoothness_grid_ny", 18));
        lower_quantile=Float64(get(cfg, "smoothness_lower_quantile", 0.01)),
        upper_quantile=Float64(get(cfg, "smoothness_upper_quantile", 0.99)),
        padding_fraction=Float64(get(cfg, "smoothness_padding_fraction", 0.04)),
    )
    xgrid = axes.x
    ygrid = axes.y
    pairs_a = Vector{NTuple{2,Float32}}()
    pairs_b = Vector{NTuple{2,Float32}}()
    inv_dx_list = Float32[]
    inv_dy_list = Float32[]
    # x-direction pairs: (x_i, y_j) -> (x_{i+1}, y_j)
    for iy in eachindex(ygrid), ix in 1:(length(xgrid) - 1)
        push!(pairs_a, (Float32(xgrid[ix]), Float32(ygrid[iy])))
        push!(pairs_b, (Float32(xgrid[ix + 1]), Float32(ygrid[iy])))
        push!(inv_dx_list, Float32(1.0 / (xgrid[ix + 1] - xgrid[ix])))
    end
    n_x_pairs = length(pairs_a)
    # y-direction pairs: (x_i, y_j) -> (x_i, y_{j+1})
    for iy in 1:(length(ygrid) - 1), ix in eachindex(xgrid)
        push!(pairs_a, (Float32(xgrid[ix]), Float32(ygrid[iy])))
        push!(pairs_b, (Float32(xgrid[ix]), Float32(ygrid[iy + 1])))
        push!(inv_dy_list, Float32(1.0 / (ygrid[iy + 1] - ygrid[iy])))
    end
    xa = hcat([[p[1], p[2]] for p in pairs_a]...)
    xb = hcat([[p[1], p[2]] for p in pairs_b]...)
    sa = Float32.(batched_model_outputs(score_wrapper, xa; batch_size=batch_size, use_gpu=false))
    sb = Float32.(batched_model_outputs(score_wrapper, xb; batch_size=batch_size, use_gpu=false))
    inv_dx = hcat(inv_dx_list...)
    inv_dy = hcat(inv_dy_list...)
    return (
        xa=Float32.(xa), xb=Float32.(xb), sa=sa, sb=sb,
        n_x_pairs=n_x_pairs, inv_dx=inv_dx, inv_dy=inv_dy,
    )
end

function delta_m_divergence_penalty(model, div_ctx)
    comps_a = mobility_components(model, div_ctx.xa, div_ctx.sa)
    comps_b = mobility_components(model, div_ctx.xb, div_ctx.sb)
    n_x = div_ctx.n_x_pairs
    inv_dx = reshape(div_ctx.inv_dx, 1, :)
    inv_dy = reshape(div_ctx.inv_dy, 1, :)
    # x-direction derivatives: ∂M₁₁/∂x₁ and ∂M₂₁/∂x₁
    dM11_dx = (comps_b.m11[1:n_x] .- comps_a.m11[1:n_x]) .* vec(inv_dx)
    dM21_dx = (comps_b.m21[1:n_x] .- comps_a.m21[1:n_x]) .* vec(inv_dx)
    # y-direction derivatives: ∂M₁₂/∂x₂ and ∂M₂₂/∂x₂
    dM12_dy = (comps_b.m12[(n_x + 1):end] .- comps_a.m12[(n_x + 1):end]) .* vec(inv_dy)
    dM22_dy = (comps_b.m22[(n_x + 1):end] .- comps_a.m22[(n_x + 1):end]) .* vec(inv_dy)
    # Divergence row 1 ≈ ∂M₁₁/∂x₁, Divergence row 2 ≈ ∂M₂₂/∂x₂
    # We penalise each direction separately since the grid pairs don't align for cross-sums
    return mean(abs2, dM11_dx) + mean(abs2, dM21_dx) + mean(abs2, dM12_dy) + mean(abs2, dM22_dy)
end

function delta_m_stage_schedule(delta_cfg::Dict{String,Any})
    if haskey(delta_cfg, "schedule")
        return delta_cfg["schedule"]
    end
    return [
        Dict("name" => "stabilize", "epochs" => Int(delta_cfg["epochs"]), "learning_rate" => Float64(delta_cfg["learning_rate"]), "coordinate_weight" => 1.0, "quadratic_weight" => 1.0),
    ]
end

function delta_m_candidates(delta_cfg::Dict{String,Any})
    if haskey(delta_cfg, "candidates")
        return delta_cfg["candidates"]
    end
    return [
        Dict(
            "name" => "default",
            "anchor_regularization" => Float64(get(delta_cfg, "magnitude_regularization", 1.0e-4)),
            "smoothness_regularization" => Float64(get(delta_cfg, "smoothness_regularization", 0.0)),
            "residual_scale_factor" => Float64(get(delta_cfg, "residual_scale_factor", 1.0)),
        ),
    ]
end

function delta_m_residual_scale(base_residual_scales::AbstractVector,
                                delta_cfg::Dict{String,Any},
                                candidate::Dict{String,Any})
    residual_scale = Float32(get(candidate, "residual_scale_factor", get(delta_cfg, "residual_scale_factor", 1.0))) .* Float32.(base_residual_scales)
    multipliers = if haskey(candidate, "residual_scale_multipliers")
        candidate["residual_scale_multipliers"]
    elseif haskey(delta_cfg, "residual_scale_multipliers")
        delta_cfg["residual_scale_multipliers"]
    else
        nothing
    end
    multipliers === nothing && return residual_scale
    channel_multipliers = Float32.(collect(multipliers))
    length(channel_multipliers) == length(residual_scale) ||
        error("`delta_m.residual_scale_multipliers` must have length $(length(residual_scale)); got $(length(channel_multipliers)).")
    return residual_scale .* channel_multipliers
end

function delta_m_posthoc_calibration_candidates(delta_cfg::Dict{String,Any})
    parse_bool(get(delta_cfg, "enable_posthoc_calibration", false)) || return NamedTuple[]
    raw_candidates = get(delta_cfg, "posthoc_residual_scale_candidates", nothing)
    raw_candidates === nothing && return NamedTuple[]
    candidates = NamedTuple[]
    for (idx, raw_candidate) in enumerate(raw_candidates)
        raw_candidate isa Dict{String,Any} || error("Each `delta_m.posthoc_residual_scale_candidates` entry must be a table.")
        haskey(raw_candidate, "residual_scale_multipliers") ||
            error("Missing `residual_scale_multipliers` in `delta_m.posthoc_residual_scale_candidates[$idx]`.")
        push!(candidates, (
            name=String(get(raw_candidate, "name", "posthoc_$(idx)")),
            multipliers=Float32.(collect(raw_candidate["residual_scale_multipliers"])),
        ))
    end
    return candidates
end

function project_delta_m_model(model::DeltaMMobilityField,
                               multipliers::AbstractVector)
    length(multipliers) == length(model.residual_scale) ||
        error("Residual-scale multiplier length mismatch: expected $(length(model.residual_scale)), got $(length(multipliers)).")
    residual_scale = Float32.(model.residual_scale) .* Float32.(collect(multipliers))
    return DeltaMMobilityField(
        model.backbone,
        model.state_mean,
        model.state_scale,
        model.feature_mean,
        model.feature_scale,
        model.quad_center,
        model.phi_baseline,
        model.theta_baseline,
        residual_scale,
        model.eps_floor,
    )
end

function prepare_delta_m_selection_baseline(data,
                                            observables,
                                            lag_steps::AbstractVector{<:Integer},
                                            rollout_cfg::Dict{String,Any},
                                            score_wrapper,
                                            phi_sigma,
                                            paths,
                                            config)
    lag_steps_full = vcat(0, Int.(lag_steps))
    lag_times = Float64.(lag_steps_full) .* data.sample_dt
    x0 = sample_initial_conditions(data.uncorrelated, Int(rollout_cfg["n_ensembles"]), Int(rollout_cfg["seed"]))
    integration_cfg = config["integration"]
    baseline_device = parse_bool(get(integration_cfg, "use_gpu", true)) && CUDA.functional() ? :gpu : :cpu
    baseline_device === :gpu && set_gpu!(Int(get(config["training"], "gpu_id", 1)))
    baseline_snapshots = evolve_affine_score_langevin_snapshots(
        score_wrapper,
        x0,
        phi_sigma.Phi,
        phi_sigma.Sigma;
        dt=Float64(rollout_cfg["dt"]),
        n_steps=Int(rollout_cfg["n_steps"]),
        burn_in=Int(rollout_cfg["burn_in"]),
        resolution=Int(rollout_cfg["resolution"]),
        device=baseline_device,
        seed=Int(rollout_cfg["seed"]),
        progress=false,
        progress_desc="δM selection baseline",
    )
    results = Dict{String,Any}()
    for obs in observables
        C_obs = empirical_observable_correlation_tensor(obs, data.correlated, lag_steps_full)
        C_baseline = empirical_observable_correlation_tensor(obs, baseline_snapshots, lag_steps_full)
        results[obs.name] = (
            C_obs=C_obs,
            C_baseline=C_baseline,
            relative_rmse=relative_frobenius_rmse(C_obs, C_baseline),
        )
    end
    return (
        x0=x0,
        lag_steps=lag_steps_full,
        lag_times=lag_times,
        baseline_snapshots=baseline_snapshots,
        baseline_results=results,
    )
end

function evaluate_delta_m_candidate_rollout(delta_model,
                                            score_wrapper,
                                            data,
                                            observables,
                                            rollout_cfg::Dict{String,Any},
                                            baseline_bundle;
                                            divergence_mode::Symbol=:finite_difference)
    axes_info = build_field_axes(
        data.uncorrelated,
        Int(rollout_cfg["grid_nx"]),
        Int(rollout_cfg["grid_ny"]),
        Float64(rollout_cfg["grid_margin_fraction"]),
    )
    grid_data = build_delta_m_grids(delta_model, score_wrapper, axes_info; batch_size=8192, use_gpu=false, divergence_mode=divergence_mode)
    lo = min(first(grid_data.xgrid), first(grid_data.ygrid))
    hi = max(last(grid_data.xgrid), last(grid_data.ygrid))
    snapshots = ScoreEstimation.evolve_interpolated_langevin_snapshots(
        baseline_bundle.x0,
        grid_data.xgrid,
        grid_data.ygrid,
        grid_data.drift_grid,
        grid_data.sigma_grid;
        dt=Float64(rollout_cfg["dt"]),
        n_steps=Int(rollout_cfg["n_steps"]),
        burn_in=Int(rollout_cfg["burn_in"]),
        resolution=Int(rollout_cfg["resolution"]),
        seed=Int(rollout_cfg["seed"]),
        boundary=(lo, hi),
        progress=false,
        progress_desc="δM selection rollout",
    )
    results = Dict{String,Any}()
    for obs in observables
        C_model = empirical_observable_correlation_tensor(obs, snapshots, baseline_bundle.lag_steps)
        ref = baseline_bundle.baseline_results[obs.name]
        results[obs.name] = (
            relative_rmse=relative_frobenius_rmse(ref.C_obs, C_model),
            improvement=1.0 - relative_frobenius_rmse(ref.C_obs, C_model) / max(ref.relative_rmse, eps(Float64)),
        )
    end
    return (
        snapshots=snapshots,
        coordinates=results["Coordinates"],
        quadratics=results["Centered Quadratics"],
    )
end

function candidate_rank_tuple(candidate_metrics, thresholds)
    coord_ok = candidate_metrics.rollout.coordinates.relative_rmse <= thresholds.coord_limit
    quad_ok = candidate_metrics.rollout.quadratics.improvement >= thresholds.quad_target
    mean_ok = candidate_metrics.mean_delta_ratio <= thresholds.mean_delta_ratio_max
    psd_ok = candidate_metrics.min_lambda_samples > 0.0
    feasible = coord_ok && quad_ok && mean_ok && psd_ok
    return (
        feasible ? 0 : 1,
        -candidate_metrics.rollout.quadratics.improvement,
        -candidate_metrics.rollout.coordinates.improvement,
        candidate_metrics.residual_aggregate_rmse,
        candidate_metrics.mean_delta_ratio,
    )
end

function posthoc_calibration_rank_tuple(calibration_metrics, thresholds)
    coord_ok = calibration_metrics.rollout.coordinates.relative_rmse <= thresholds.coord_limit
    quad_ok = calibration_metrics.rollout.quadratics.improvement >= thresholds.quad_target
    mean_ok = calibration_metrics.mean_delta_ratio <= thresholds.mean_delta_ratio_max
    psd_ok = calibration_metrics.min_lambda_samples > 0.0
    feasible = coord_ok && quad_ok && mean_ok && psd_ok
    return (
        feasible ? 0 : 1,
        -calibration_metrics.rollout.quadratics.improvement,
        -calibration_metrics.rollout.coordinates.improvement,
        calibration_metrics.symmetric_activity,
        calibration_metrics.mean_delta_ratio,
    )
end

function run_delta_m(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :delta_m, force)
        return Dict("status" => "skipped", "delta_m_png" => paths.delta_m_png)
    end

    require_artifact(paths.dataset_h5, :generate_data, :delta_m)
    require_artifact(paths.score_model_bson, :train_score, :delta_m)
    require_artifact(paths.phi_sigma_h5, :estimate_phi_sigma, :delta_m)
    require_artifact(paths.observed_residuals_h5, :observed_residuals, :delta_m)
    require_artifact(paths.joint_score_model_bson, :joint_score, :delta_m)

    data = load_dataset(paths.dataset_h5)
    delta_cfg = config["delta_m"]
    observable_cfg = get(config, "delta_m_observables", Dict{String,Any}())
    observables = build_delta_m_observable_specs(data.uncorrelated, observable_cfg)
    rollout_observables = build_base_observable_specs(data.uncorrelated)
    lag_steps = build_lag_steps(
        data,
        Float64(delta_cfg["tD_override"]),
        Int(delta_cfg["lag_step_stride"]);
        include_zero=false,
    )
    lag_times = Float64.(lag_steps) .* data.sample_dt
    targets = read_observed_residual_targets(paths.observed_residuals_h5, observables, lag_steps)
    conditional_wrapper = build_conditional_score_wrapper(paths)
    score_wrapper, score_meta = build_score_wrapper(paths.score_model_bson)
    phi_sigma = h5open(paths.phi_sigma_h5, "r") do file
        (
            Phi=Float32.(read(file["Phi"])),
            Sigma=Float32.(read(file["Sigma"])),
        )
    end

    gpu_ids = available_gpu_ids()
    use_gpu = parse_bool(get(delta_cfg, "use_gpu", true)) && !isempty(gpu_ids)
    gpu_id = Int(get(delta_cfg, "gpu_id", isempty(gpu_ids) ? 0 : first(gpu_ids)))
    if use_gpu
        set_gpu!(gpu_id)
    end

    rng = MersenneTwister(Int(delta_cfg["seed"]))
    verbose = parse_bool(get(delta_cfg, "verbose", true))
    lag_lookup = Dict(step => idx for (idx, step) in enumerate(Int.(lag_steps)))
    lag_buckets = delta_m_lag_buckets(lag_steps, lag_times)
    eval_pairs = prepare_delta_m_eval_pairs(
        data.correlated,
        lag_steps;
        max_pairs=Int(delta_cfg["eval_pairs_per_lag"]),
        seed=Int(delta_cfg["seed"]) + 7,
    )
    feature_mean, feature_scale = mobility_feature_statistics(
        data.uncorrelated,
        score_wrapper,
        score_meta["mean"],
        score_meta["scale"];
        sample_count=Int(get(delta_cfg, "feature_stat_samples", 50_000)),
        batch_size=Int(get(delta_cfg, "batch_size", 4096)),
        seed=Int(delta_cfg["seed"]) + 11,
    )
    smoothness_ctx_cpu = build_smoothness_context(data.uncorrelated, score_wrapper, delta_cfg; batch_size=Int(get(delta_cfg, "batch_size", 4096)))
    divergence_ctx_cpu = build_divergence_context(data.uncorrelated, score_wrapper, delta_cfg; batch_size=Int(get(delta_cfg, "batch_size", 4096)))
    schedule = delta_m_stage_schedule(delta_cfg)
    candidates = delta_m_candidates(delta_cfg)
    divergence_mode = resolve_divergence_mode(get(delta_cfg, "divergence_mode", "autodiff"))
    base_residual_scales = Float32.(get(delta_cfg, "base_residual_scales", [0.75, 0.35, 0.75, 0.35]))
    omega_baseline = 0.5f0 * (phi_sigma.Phi[1, 2] - phi_sigma.Phi[2, 1])
    λmean = Float32(get(delta_cfg, "mean_regularization", 0.0))
    λpsd = Float32(get(delta_cfg, "psd_margin_regularization", 0.0))
    λdiv = Float32(get(delta_cfg, "divergence_regularization", 0.0))
    psd_floor = Float32(get(delta_cfg, "psd_margin_floor", 0.0))
    use_global_target_scale = parse_bool(get(delta_cfg, "use_global_target_scale", false))
    global_target_scales = Dict{String,Float32}()
    for obs in observables
        target_tensor = targets[obs.name].E_obs
        global_target_scales[obs.name] = Float32(max(mean(abs2, target_tensor), 1.0f-6))
    end
    verbose && use_global_target_scale && println("[delta_m] Using global target scales: ", Dict(k => round(v, sigdigits=3) for (k, v) in global_target_scales))
    verbose && λdiv > 0 && println("[delta_m] Divergence regularization λ_div = $(λdiv)")
    baseline_bundle = prepare_delta_m_selection_baseline(
        data,
        rollout_observables,
        resolve_delta_m_rollout_lag_steps(data, lag_steps, config["delta_m_rollout"], Int(delta_cfg["lag_step_stride"])),
        config["delta_m_rollout"],
        score_wrapper,
        phi_sigma,
        paths,
        config,
    )
    coord_tol = Float64(get(delta_cfg, "candidate_coordinate_tolerance_fraction", 0.0))
    thresholds = (
        coord_limit=baseline_bundle.baseline_results["Coordinates"].relative_rmse * (1.0 + coord_tol),
        quad_target=Float64(get(delta_cfg, "candidate_quadratics_improvement_target", 0.05)),
        mean_delta_ratio_max=Float64(get(delta_cfg, "candidate_mean_delta_ratio_max", 0.25)),
    )

    candidate_records = NamedTuple[]
    best_candidate = nothing
    best_rank = nothing

    for (candidate_idx, candidate) in enumerate(candidates)
        candidate_name = String(get(candidate, "name", "candidate_$(candidate_idx)"))
        λanchor = Float32(get(candidate, "anchor_regularization", get(delta_cfg, "magnitude_regularization", 0.0)))
        λsmooth = Float32(get(candidate, "smoothness_regularization", get(delta_cfg, "smoothness_regularization", 0.0)))
        residual_scale = delta_m_residual_scale(base_residual_scales, delta_cfg, candidate)
        model = DeltaMMobilityField(
            Float32.(score_meta["mean"]),
            Float32.(score_meta["scale"]),
            feature_mean,
            feature_scale,
            Float32.(score_meta["mean"]),
            Int.(delta_cfg["neurons"]),
            phi_sigma.Phi,
            phi_sigma.Sigma,
            omega_baseline,
            residual_scale;
            eps_floor=Float64(delta_cfg["eps_floor"]),
        )
        model = Flux.f32(model)
        model = use_gpu ? ScoreEstimation._move_to_gpu(model, CUDA) : model
        conditional_eval = use_gpu ? ScoreEstimation._move_to_gpu(deepcopy(conditional_wrapper), CUDA) : deepcopy(conditional_wrapper)
        score_eval = use_gpu ? ScoreEstimation._move_to_gpu(deepcopy(score_wrapper), CUDA) : deepcopy(score_wrapper)
        Flux.testmode!(conditional_eval)
        Flux.testmode!(score_eval)
        smoothness_ctx = (
            xa=ScoreEstimation._to_device(smoothness_ctx_cpu.xa; use_gpu=use_gpu),
            xb=ScoreEstimation._to_device(smoothness_ctx_cpu.xb; use_gpu=use_gpu),
            sa=ScoreEstimation._to_device(smoothness_ctx_cpu.sa; use_gpu=use_gpu),
            sb=ScoreEstimation._to_device(smoothness_ctx_cpu.sb; use_gpu=use_gpu),
        )
        divergence_ctx = (
            xa=ScoreEstimation._to_device(divergence_ctx_cpu.xa; use_gpu=use_gpu),
            xb=ScoreEstimation._to_device(divergence_ctx_cpu.xb; use_gpu=use_gpu),
            sa=ScoreEstimation._to_device(divergence_ctx_cpu.sa; use_gpu=use_gpu),
            sb=ScoreEstimation._to_device(divergence_ctx_cpu.sb; use_gpu=use_gpu),
            n_x_pairs=divergence_ctx_cpu.n_x_pairs,
            inv_dx=ScoreEstimation._to_device(divergence_ctx_cpu.inv_dx; use_gpu=use_gpu),
            inv_dy=ScoreEstimation._to_device(divergence_ctx_cpu.inv_dy; use_gpu=use_gpu),
        )

        losses = Float32[]
        mean_penalties = Float32[]
        mag_penalties = Float32[]
        stage_rollout = NamedTuple[]
        max_batches = Int(delta_cfg["max_batches_per_epoch"])
        t0 = time()

        for (stage_idx, stage_cfg) in enumerate(schedule)
            if stage_idx == 3 && !isempty(stage_rollout) && last(stage_rollout).coordinates_relative_rmse > thresholds.coord_limit
                push!(stage_rollout, (stage=stage_idx, skipped=true, coordinates_relative_rmse=last(stage_rollout).coordinates_relative_rmse, quadratics_relative_rmse=last(stage_rollout).quadratics_relative_rmse))
                continue
            end

            stage_epochs = Int(get(stage_cfg, "epochs", delta_cfg["epochs"]))
            stage_lr = Float32(get(stage_cfg, "learning_rate", delta_cfg["learning_rate"]))
            coord_weight = Float32(get(stage_cfg, "coordinate_weight", 1.0))
            poly_weight = Float32(get(stage_cfg, "polynomial_weight", get(stage_cfg, "quadratic_weight", 1.0)))
            rbf_weight = Float32(get(stage_cfg, "rbf_weight", poly_weight))
            opt_state = Flux.setup(Flux.Adam(stage_lr), model)
            verbose && println("[delta_m] candidate=$(candidate_name) stage $(stage_idx)/$(length(schedule)) epochs=$(stage_epochs) lr=$(stage_lr)")
            verbose && (progress = ScoreEstimation.ProgressMeter.Progress(stage_epochs; desc="δM epochs ($(candidate_name))", showspeed=false))

            for _ in 1:stage_epochs
                epoch_losses = Float32[]
                epoch_mean_penalties = Float32[]
                epoch_anchor_penalties = Float32[]
                for _ in 1:max_batches
                    batch_specs = (
                        Int(rand(rng, lag_buckets.short)),
                        Int(rand(rng, lag_buckets.medium)),
                        Int(rand(rng, lag_buckets.long)),
                    )
                    prepared = NamedTuple[]
                    for lag_step in batch_specs
                        lag_time = lag_step * data.sample_dt
                        lag_idx = lag_lookup[lag_step]
                        x0_cpu, xt_cpu = sample_lagged_pairs(data.correlated, lag_step, Int(delta_cfg["batch_size"]), rng)
                        x0 = ScoreEstimation._to_device(x0_cpu; use_gpu=use_gpu)
                        xt = ScoreEstimation._to_device(xt_cpu; use_gpu=use_gpu)
                        scond = conditional_score(conditional_eval, x0, xt, lag_time)
                        score0 = score_eval(x0)
                        phi_batches = Dict(obs.name => evaluate_observable_device(obs, xt) for obs in observables)
                        target_mats = Dict(
                            obs.name => ScoreEstimation._to_device(Float32.(targets[obs.name].E_obs[:, :, lag_idx]); use_gpu=use_gpu)
                            for obs in observables
                        )
                        push!(prepared, (x0=x0, scond=scond, score0=score0, phi_batches=phi_batches, target_mats=target_mats))
                    end

                    loss_value, grads = Flux.withgradient(model) do mobility_model
                        local_loss = zero(Float32)
                        mean_pen = zero(Float32)
                        anchor_pen = zero(Float32)
                        psd_pen = zero(Float32)
                        for batch in prepared
                            comps = mobility_components(mobility_model, batch.x0, batch.score0)
                            eff = effective_delta_score(comps, batch.scond)
                            for obs in observables
                                phi_xt = batch.phi_batches[obs.name]
                                estimate = (phi_xt * transpose(eff)) ./ size(batch.x0, 2)
                                target = batch.target_mats[obs.name]
                                weight = obs.group === :coordinate ? coord_weight : (obs.group === :rbf ? rbf_weight : poly_weight)
                                scale = use_global_target_scale ? global_target_scales[obs.name] : max(mean(abs2, target), 1.0f-6)
                                local_loss += weight * mean(abs2, estimate .- target) / scale
                            end
                            mean_pen += delta_m_mean_penalty(comps)
                            anchor_pen += delta_m_anchor_penalty(comps)
                            psd_pen += delta_m_psd_margin_penalty(comps, psd_floor)
                        end
                        local_loss /= Float32(length(prepared))
                        mean_pen /= Float32(length(prepared))
                        anchor_pen /= Float32(length(prepared))
                        psd_pen /= Float32(length(prepared))
                        smooth_pen = delta_m_smoothness_penalty(mobility_model, smoothness_ctx.xa, smoothness_ctx.sa, smoothness_ctx.xb, smoothness_ctx.sb)
                        div_pen = λdiv > 0 ? delta_m_divergence_penalty(mobility_model, divergence_ctx) : zero(Float32)
                        return local_loss + λmean * mean_pen + λanchor * anchor_pen + λsmooth * smooth_pen + λpsd * psd_pen + λdiv * div_pen
                    end

                    Flux.update!(opt_state, model, grads[1])
                    push!(epoch_losses, Float32(loss_value))
                    ref_batch = first(prepared)
                    ref_comps = mobility_components(model, ref_batch.x0, ref_batch.score0)
                    push!(epoch_mean_penalties, Float32(delta_m_mean_penalty(ref_comps)))
                    push!(epoch_anchor_penalties, Float32(delta_m_anchor_penalty(ref_comps)))
                end

                push!(losses, mean(epoch_losses))
                push!(mean_penalties, mean(epoch_mean_penalties))
                push!(mag_penalties, mean(epoch_anchor_penalties))
                verbose && ScoreEstimation.ProgressMeter.next!(progress)
            end
            verbose && ScoreEstimation.ProgressMeter.finish!(progress)

            if stage_idx == min(2, length(schedule))
                stage_model_cpu = Flux.cpu(model)
                rollout_mid = evaluate_delta_m_candidate_rollout(stage_model_cpu, score_wrapper, data, rollout_observables, config["delta_m_rollout"], baseline_bundle; divergence_mode=divergence_mode)
                push!(stage_rollout, (stage=stage_idx, skipped=false, coordinates_relative_rmse=rollout_mid.coordinates.relative_rmse, quadratics_relative_rmse=rollout_mid.quadratics.relative_rmse))
            end
        end

        elapsed = time() - t0
        model_cpu = Flux.cpu(model)
        residual_results = evaluate_delta_m_residuals(
            model_cpu,
            score_wrapper,
            conditional_wrapper,
            observables,
            targets,
            eval_pairs,
            lag_times,
            lag_steps;
            use_gpu=use_gpu,
            seed=Int(delta_cfg["seed"]) + candidate_idx,
        )
        sample_summary = summarize_delta_m_samples(
            model_cpu,
            score_wrapper,
            sample_initial_conditions(data.uncorrelated, Int(delta_cfg["check_samples"]), Int(delta_cfg["seed"]) + 91 + candidate_idx),
        )
        rollout_metrics = evaluate_delta_m_candidate_rollout(model_cpu, score_wrapper, data, rollout_observables, config["delta_m_rollout"], baseline_bundle; divergence_mode=divergence_mode)
        residual_map = Dict(result.name => result.relative_rmse for result in residual_results)
        residual_weight_total = sum(length(result.output_labels) for result in residual_results)
        residual_aggregate_rmse = sum(length(result.output_labels) * result.relative_rmse for result in residual_results) / max(residual_weight_total, 1)
        candidate_metrics = (
            name=candidate_name,
            model=model_cpu,
            losses=copy(losses),
            mean_penalties=copy(mean_penalties),
            mag_penalties=copy(mag_penalties),
            elapsed=elapsed,
            sample_summary=sample_summary,
            residual_results=residual_results,
            residual_coords_rmse=residual_map["Coordinates"],
            residual_quads_rmse=residual_map["Centered Quadratics"],
            residual_aggregate_rmse=residual_aggregate_rmse,
            rollout=rollout_metrics,
            residual_scale=Float32.(model_cpu.residual_scale),
            mean_delta_ratio=sample_summary.mean_delta_norm / max(norm(phi_sigma.Phi), eps(Float64)),
            min_lambda_samples=sample_summary.min_lambda,
        )
        push!(candidate_records, candidate_metrics)
        rank = candidate_rank_tuple(candidate_metrics, thresholds)
        if best_rank === nothing || rank < best_rank
            best_rank = rank
            best_candidate = candidate_metrics
        end
    end

    best_candidate === nothing && error("No δM candidate could be selected.")
    selected_training_candidate_name = best_candidate.name
    selected_posthoc_candidate_name = nothing
    selected_posthoc_multipliers = Float32[]
    posthoc_candidate_records = NamedTuple[]

    calibration_candidates = delta_m_posthoc_calibration_candidates(delta_cfg)
    if !isempty(calibration_candidates)
        calibration_best = nothing
        calibration_best_rank = nothing
        for calibration_candidate in calibration_candidates
            calibrated_model = project_delta_m_model(best_candidate.model, calibration_candidate.multipliers)
            sample_summary = summarize_delta_m_samples(
                calibrated_model,
                score_wrapper,
                sample_initial_conditions(data.uncorrelated, Int(delta_cfg["check_samples"]), Int(delta_cfg["seed"]) + 211),
            )
            rollout_metrics = evaluate_delta_m_candidate_rollout(
                calibrated_model,
                score_wrapper,
                data,
                rollout_observables,
                config["delta_m_rollout"],
                baseline_bundle;
                divergence_mode=divergence_mode,
            )
            calibration_metrics = (
                name=calibration_candidate.name,
                multipliers=Float32.(calibration_candidate.multipliers),
                model=calibrated_model,
                sample_summary=sample_summary,
                rollout=rollout_metrics,
                residual_scale=Float32.(calibrated_model.residual_scale),
                mean_delta_ratio=sample_summary.mean_delta_norm / max(norm(phi_sigma.Phi), eps(Float64)),
                min_lambda_samples=sample_summary.min_lambda,
                symmetric_activity=sum(abs, Float64.(calibration_candidate.multipliers[1:min(3, end)])),
            )
            push!(posthoc_candidate_records, calibration_metrics)
            rank = posthoc_calibration_rank_tuple(calibration_metrics, thresholds)
            if calibration_best_rank === nothing || rank < calibration_best_rank
                calibration_best_rank = rank
                calibration_best = calibration_metrics
            end
        end

        calibration_best === nothing && error("No posthoc δM calibration candidate could be selected.")
        selected_posthoc_candidate_name = calibration_best.name
        selected_posthoc_multipliers = Float32.(calibration_best.multipliers)
        residual_results = evaluate_delta_m_residuals(
            calibration_best.model,
            score_wrapper,
            conditional_wrapper,
            observables,
            targets,
            eval_pairs,
            lag_times,
            lag_steps;
            use_gpu=use_gpu,
            seed=Int(delta_cfg["seed"]) + 313,
        )
        residual_map = Dict(result.name => result.relative_rmse for result in residual_results)
        residual_weight_total = sum(length(result.output_labels) for result in residual_results)
        residual_aggregate_rmse = sum(length(result.output_labels) * result.relative_rmse for result in residual_results) / max(residual_weight_total, 1)
        best_candidate = (
            name=string(selected_training_candidate_name, " + ", selected_posthoc_candidate_name),
            model=calibration_best.model,
            losses=best_candidate.losses,
            mean_penalties=best_candidate.mean_penalties,
            mag_penalties=best_candidate.mag_penalties,
            elapsed=best_candidate.elapsed,
            sample_summary=calibration_best.sample_summary,
            residual_results=residual_results,
            residual_coords_rmse=residual_map["Coordinates"],
            residual_quads_rmse=residual_map["Centered Quadratics"],
            residual_aggregate_rmse=residual_aggregate_rmse,
            rollout=calibration_best.rollout,
            residual_scale=Float32.(calibration_best.residual_scale),
            mean_delta_ratio=calibration_best.mean_delta_ratio,
            min_lambda_samples=calibration_best.min_lambda_samples,
        )
    end

    model_cpu = best_candidate.model
    metadata = Dict(
        "training_time_seconds" => best_candidate.elapsed,
        "lag_steps" => Int.(lag_steps),
        "lag_times" => Float64.(lag_times),
        "tD" => Float64(delta_cfg["tD_override"]),
        "mean" => Float32.(score_meta["mean"]),
        "scale" => Float32.(score_meta["scale"]),
        "eps_floor" => Float64(delta_cfg["eps_floor"]),
        "phi_baseline" => Float32.(phi_sigma.Phi),
        "selected_candidate_name" => best_candidate.name,
        "selected_training_candidate_name" => selected_training_candidate_name,
        "residual_scale" => Float32.(best_candidate.residual_scale),
        "feature_mean" => feature_mean,
        "feature_scale" => feature_scale,
        "divergence_mode" => String(divergence_mode),
    )
    if selected_posthoc_candidate_name !== nothing
        metadata["selected_posthoc_candidate_name"] = selected_posthoc_candidate_name
        metadata["posthoc_residual_scale_multipliers"] = Float32.(selected_posthoc_multipliers)
    end
    save_model(model_cpu, paths.delta_m_model_bson; metadata=metadata)

    grid_axes = build_field_axes(
        data.uncorrelated,
        Int(delta_cfg["grid_nx"]),
        Int(delta_cfg["grid_ny"]),
        Float64(delta_cfg["grid_margin_fraction"]),
    )
    grid_data = build_delta_m_grids(model_cpu, score_wrapper, grid_axes; batch_size=Int(delta_cfg["batch_size"]), use_gpu=false, divergence_mode=divergence_mode)
    fig = delta_m_training_figure(best_candidate.losses, best_candidate.mean_penalties, best_candidate.mag_penalties, grid_data, best_candidate.residual_results, lag_times)
    save(paths.delta_m_png, fig, px_per_unit=2)
    save(paths.delta_m_pdf, fig)

    h5open(paths.delta_m_h5, "w") do file
        file["selected_candidate_name"] = best_candidate.name
        file["selected_training_candidate_name"] = selected_training_candidate_name
        file["losses"] = Float32.(best_candidate.losses)
        file["mean_penalties"] = Float32.(best_candidate.mean_penalties)
        file["magnitude_penalties"] = Float32.(best_candidate.mag_penalties)
        file["lag_steps"] = Int.(lag_steps)
        file["lag_times"] = Float64.(lag_times)
        file["xgrid"] = Float64.(grid_data.xgrid)
        file["ygrid"] = Float64.(grid_data.ygrid)
        file["delta_grid"] = grid_data.delta_grid
        file["omega_grid"] = grid_data.omega
        file["lambda_min_grid"] = grid_data.lambda_min
        file["drift_grid"] = grid_data.drift_grid
        file["score_term_grid"] = grid_data.score_term_grid
        file["divergence_grid"] = grid_data.divergence_grid
        file["sigma_grid"] = grid_data.sigma_grid
        file["divergence_mode"] = String(divergence_mode)
        file["mean_delta"] = best_candidate.sample_summary.mean_delta
        file["mean_delta_norm"] = best_candidate.sample_summary.mean_delta_norm
        file["min_lambda_samples"] = best_candidate.sample_summary.min_lambda
        file["skew_error"] = best_candidate.sample_summary.skew_error
        file["residual_scale"] = Float32.(best_candidate.residual_scale)
        if selected_posthoc_candidate_name !== nothing
            file["selected_posthoc_candidate_name"] = selected_posthoc_candidate_name
            file["posthoc_residual_scale_multipliers"] = Float32.(selected_posthoc_multipliers)
        end
        file["coordinates_rollout_relative_rmse"] = best_candidate.rollout.coordinates.relative_rmse
        file["quadratics_rollout_relative_rmse"] = best_candidate.rollout.quadratics.relative_rmse
        file["coordinates_rollout_improvement"] = best_candidate.rollout.coordinates.improvement
        file["quadratics_rollout_improvement"] = best_candidate.rollout.quadratics.improvement
        file["aggregate_relative_rmse"] = best_candidate.residual_aggregate_rmse
        for result in best_candidate.residual_results
            group = create_group(file, slugify(result.name))
            group["E_obs"] = result.E_obs
            group["E_model"] = result.E_model
            group["relative_error"] = result.relative_error
            group["relative_rmse"] = result.relative_rmse
            for (label_idx, label) in enumerate(result.output_labels)
                group["output_label_$(label_idx)"] = label
            end
        end
        candidates_group = create_group(file, "candidates")
        for (idx, candidate_metrics) in enumerate(candidate_records)
            group = create_group(candidates_group, "candidate_$(idx)")
            group["name"] = candidate_metrics.name
            group["coordinates_residual_rmse"] = candidate_metrics.residual_coords_rmse
            group["quadratics_residual_rmse"] = candidate_metrics.residual_quads_rmse
            group["coordinates_rollout_relative_rmse"] = candidate_metrics.rollout.coordinates.relative_rmse
            group["quadratics_rollout_relative_rmse"] = candidate_metrics.rollout.quadratics.relative_rmse
            group["coordinates_rollout_improvement"] = candidate_metrics.rollout.coordinates.improvement
            group["quadratics_rollout_improvement"] = candidate_metrics.rollout.quadratics.improvement
            group["residual_aggregate_rmse"] = candidate_metrics.residual_aggregate_rmse
            group["mean_delta_ratio"] = candidate_metrics.mean_delta_ratio
            group["min_lambda_samples"] = candidate_metrics.min_lambda_samples
            group["residual_scale"] = Float32.(candidate_metrics.residual_scale)
        end
        if !isempty(posthoc_candidate_records)
            posthoc_group = create_group(file, "posthoc_calibration_candidates")
            for (idx, calibration_metrics) in enumerate(posthoc_candidate_records)
                group = create_group(posthoc_group, "candidate_$(idx)")
                group["name"] = calibration_metrics.name
                group["residual_scale_multipliers"] = Float32.(calibration_metrics.multipliers)
                group["residual_scale"] = Float32.(calibration_metrics.residual_scale)
                group["coordinates_rollout_relative_rmse"] = calibration_metrics.rollout.coordinates.relative_rmse
                group["quadratics_rollout_relative_rmse"] = calibration_metrics.rollout.quadratics.relative_rmse
                group["coordinates_rollout_improvement"] = calibration_metrics.rollout.coordinates.improvement
                group["quadratics_rollout_improvement"] = calibration_metrics.rollout.quadratics.improvement
                group["mean_delta_ratio"] = calibration_metrics.mean_delta_ratio
                group["min_lambda_samples"] = calibration_metrics.min_lambda_samples
                group["symmetric_activity"] = calibration_metrics.symmetric_activity
            end
        end
    end

    rmse_map = Dict(result.name => result.relative_rmse for result in best_candidate.residual_results)
    return Dict(
        "status" => "completed",
        "delta_m_model_bson" => paths.delta_m_model_bson,
        "delta_m_h5" => paths.delta_m_h5,
        "delta_m_png" => paths.delta_m_png,
        "delta_m_pdf" => paths.delta_m_pdf,
        "training_time_seconds" => best_candidate.elapsed,
        "selected_candidate_name" => best_candidate.name,
        "coordinates_relative_rmse" => rmse_map["Coordinates"],
        "quadratics_relative_rmse" => rmse_map["Centered Quadratics"],
        "aggregate_relative_rmse" => best_candidate.residual_aggregate_rmse,
        "coordinates_rollout_relative_rmse" => best_candidate.rollout.coordinates.relative_rmse,
        "quadratics_rollout_relative_rmse" => best_candidate.rollout.quadratics.relative_rmse,
        "coordinates_rollout_improvement" => best_candidate.rollout.coordinates.improvement,
        "quadratics_rollout_improvement" => best_candidate.rollout.quadratics.improvement,
        "mean_delta_norm" => best_candidate.sample_summary.mean_delta_norm,
        "min_lambda_samples" => best_candidate.sample_summary.min_lambda,
    )
end

function run_delta_m_rollout(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :delta_m_rollout, force)
        return Dict("status" => "skipped", "delta_m_rollout_png" => paths.delta_m_rollout_png)
    end

    require_artifact(paths.dataset_h5, :generate_data, :delta_m_rollout)
    require_artifact(paths.score_model_bson, :train_score, :delta_m_rollout)
    require_artifact(paths.phi_sigma_h5, :estimate_phi_sigma, :delta_m_rollout)
    require_artifact(paths.delta_m_model_bson, :delta_m, :delta_m_rollout)

    data = load_dataset(paths.dataset_h5)
    rollout_cfg = config["delta_m_rollout"]
    observables = build_base_observable_specs(data.uncorrelated)
    score_wrapper, _ = build_score_wrapper(paths.score_model_bson)
    delta_model, delta_meta = load_model(paths.delta_m_model_bson)
    divergence_mode = resolve_divergence_mode(get(delta_meta, "divergence_mode", get(config["delta_m"], "divergence_mode", "autodiff")))
    phi_sigma = h5open(paths.phi_sigma_h5, "r") do file
        (
            Phi=Float32.(read(file["Phi"])),
            Sigma=Float32.(read(file["Sigma"])),
        )
    end
    lag_steps_base = Int.(delta_meta["lag_steps"])
    lag_steps = resolve_delta_m_rollout_lag_steps(
        data,
        lag_steps_base,
        rollout_cfg,
        Int(get(config["delta_m"], "lag_step_stride", 1)),
    )
    lag_times = Float64.(lag_steps) .* data.sample_dt

    axes_info = build_field_axes(
        data.uncorrelated,
        Int(rollout_cfg["grid_nx"]),
        Int(rollout_cfg["grid_ny"]),
        Float64(rollout_cfg["grid_margin_fraction"]),
    )
    grid_data = build_delta_m_grids(delta_model, score_wrapper, axes_info; batch_size=8192, use_gpu=false, divergence_mode=divergence_mode)
    lo = min(first(grid_data.xgrid), first(grid_data.ygrid))
    hi = max(last(grid_data.xgrid), last(grid_data.ygrid))
    x0 = sample_initial_conditions(data.uncorrelated, Int(rollout_cfg["n_ensembles"]), Int(rollout_cfg["seed"]))
    integration_cfg = config["integration"]
    baseline_device = parse_bool(get(integration_cfg, "use_gpu", true)) && CUDA.functional() ? :gpu : :cpu
    baseline_device === :gpu && set_gpu!(Int(get(config["training"], "gpu_id", 1)))
    baseline_snapshots = evolve_affine_score_langevin_snapshots(
        score_wrapper,
        x0,
        phi_sigma.Phi,
        phi_sigma.Sigma;
        dt=Float64(rollout_cfg["dt"]),
        n_steps=Int(rollout_cfg["n_steps"]),
        burn_in=Int(rollout_cfg["burn_in"]),
        resolution=Int(rollout_cfg["resolution"]),
        device=baseline_device,
        seed=Int(rollout_cfg["seed"]),
        progress=parse_bool(get(rollout_cfg, "progress", false)),
        progress_desc="Affine baseline rollout",
    )
    snapshots = ScoreEstimation.evolve_interpolated_langevin_snapshots(
        x0,
        grid_data.xgrid,
        grid_data.ygrid,
        grid_data.drift_grid,
        grid_data.sigma_grid;
        dt=Float64(rollout_cfg["dt"]),
        n_steps=Int(rollout_cfg["n_steps"]),
        burn_in=Int(rollout_cfg["burn_in"]),
        resolution=Int(rollout_cfg["resolution"]),
        seed=Int(rollout_cfg["seed"]),
        boundary=(lo, hi),
        progress=parse_bool(get(rollout_cfg, "progress", false)),
        progress_desc="δM rollout",
    )
    baseline_stationary = stationary_from_snapshots(baseline_snapshots)
    baseline_trajectory = first_trajectory(baseline_snapshots)
    stationary = stationary_from_snapshots(snapshots)
    trajectory = first_trajectory(snapshots)

    observable_results = NamedTuple[]
    baseline_results = Dict{String,Any}()
    for obs in observables
        C_obs = empirical_observable_correlation_tensor(obs, data.correlated, lag_steps)
        C_baseline = empirical_observable_correlation_tensor(obs, baseline_snapshots, lag_steps)
        C_model = empirical_observable_correlation_tensor(obs, snapshots, lag_steps)
        baseline_rel = [norm(C_baseline[:, :, k] .- C_obs[:, :, k]) / max(norm(C_obs[:, :, k]), eps(Float64)) for k in axes(C_obs, 3)]
        model_rel = [norm(C_model[:, :, k] .- C_obs[:, :, k]) / max(norm(C_obs[:, :, k]), eps(Float64)) for k in axes(C_obs, 3)]
        baseline_results[obs.name] = (
            C_model=C_baseline,
            relative_error=baseline_rel,
            relative_rmse=relative_frobenius_rmse(C_obs, C_baseline),
        )
        push!(observable_results, (
            name=obs.name,
            output_labels=obs.output_labels,
            C_obs=C_obs,
            C_model=C_model,
            relative_error=model_rel,
            relative_rmse=relative_frobenius_rmse(C_obs, C_model),
            baseline=baseline_results[obs.name],
        ))
    end

    eval_cfg = build_eval_config(data.cfg, config)
    true_diag = build_brusselator_diagnostics(eval_cfg, data.correlated, data.uncorrelated; sample_dt=data.sample_dt)
    rollout_sample_dt = Float64(rollout_cfg["dt"]) * Int(rollout_cfg["resolution"])
    baseline_diag = build_brusselator_diagnostics(eval_cfg, baseline_trajectory, baseline_stationary; sample_dt=rollout_sample_dt, x_range=true_diag.x_range, y_range=true_diag.y_range)
    model_diag = build_brusselator_diagnostics(eval_cfg, trajectory, stationary; sample_dt=Float64(rollout_cfg["dt"]) * Int(rollout_cfg["resolution"]), x_range=true_diag.x_range, y_range=true_diag.y_range)
    baseline_metrics = compare_brusselator_diagnostics(true_diag, baseline_diag; acf_window=findlast(<=(Float64(rollout_cfg["metric_time_horizon"])), true_diag.lag_axis))
    model_metrics = compare_brusselator_diagnostics(true_diag, model_diag; acf_window=findlast(<=(Float64(rollout_cfg["metric_time_horizon"])), true_diag.lag_axis))

    coord_result = only(filter(result -> result.name == "Coordinates", observable_results))
    quad_result = only(filter(result -> result.name == "Centered Quadratics", observable_results))
    coord_baseline_rmse = coord_result.baseline.relative_rmse
    quad_baseline_rmse = quad_result.baseline.relative_rmse
    coord_improvement = 1.0 - coord_result.relative_rmse / max(coord_baseline_rmse, eps(Float64))
    quad_improvement = 1.0 - quad_result.relative_rmse / max(quad_baseline_rmse, eps(Float64))
    delta_summary = isfile(paths.delta_m_h5) ? h5open(paths.delta_m_h5, "r") do file
        Dict(
            "selected_candidate_name" => read(file["selected_candidate_name"]),
            "coordinates_residual_rmse" => read(file["coordinates/relative_rmse"]),
            "quadratics_residual_rmse" => read(file["centered_quadratics/relative_rmse"]),
        )
    end : Dict{String,Any}()
    summary_lines = [
        "Coordinates rel. RMSE = $(round(coord_result.relative_rmse, digits=5))",
        "Coordinates baseline rel. RMSE = $(round(coord_baseline_rmse, digits=5))",
        "Coordinates improvement = $(round(100 * coord_improvement, digits=2))%",
        "Quadratics rel. RMSE = $(round(quad_result.relative_rmse, digits=5))",
        "Quadratics baseline rel. RMSE = $(round(quad_baseline_rmse, digits=5))",
        "Quadratics improvement = $(round(100 * quad_improvement, digits=2))%",
        "PDF KLxy baseline = $(round(baseline_metrics.kl_xy, digits=5))",
        "PDF KLxy δM = $(round(model_metrics.kl_xy, digits=5))",
        "ACF RMSE x baseline = $(round(baseline_metrics.rmse_acf_x, digits=5))",
        "ACF RMSE x δM = $(round(model_metrics.rmse_acf_x, digits=5))",
        "ACF RMSE y baseline = $(round(baseline_metrics.rmse_acf_y, digits=5))",
        "ACF RMSE y δM = $(round(model_metrics.rmse_acf_y, digits=5))",
    ]
    if !isempty(delta_summary)
        pushfirst!(summary_lines, "selected δM candidate = $(delta_summary["selected_candidate_name"])")
        push!(summary_lines, "Residual RMSE coords = $(round(delta_summary["coordinates_residual_rmse"], digits=5))")
        push!(summary_lines, "Residual RMSE quads = $(round(delta_summary["quadratics_residual_rmse"], digits=5))")
    end

    fig = delta_m_rollout_figure(observable_results, lag_times, true_diag, baseline_diag, model_diag, summary_lines)
    save(paths.delta_m_rollout_png, fig, px_per_unit=2)
    save(paths.delta_m_rollout_pdf, fig)

    h5open(paths.delta_m_rollout_h5, "w") do file
        file["lag_steps"] = Int.(lag_steps)
        file["lag_times"] = Float64.(lag_times)
        file["divergence_mode"] = String(divergence_mode)
        file["baseline_trajectory"] = baseline_trajectory
        file["baseline_stationary"] = baseline_stationary
        file["trajectory"] = trajectory
        file["stationary"] = stationary
        file["coordinates_relative_rmse"] = coord_result.relative_rmse
        file["coordinates_baseline_relative_rmse"] = coord_baseline_rmse
        file["quadratics_relative_rmse"] = quad_result.relative_rmse
        file["quadratics_baseline_relative_rmse"] = quad_baseline_rmse
        file["coordinates_improvement"] = coord_improvement
        file["quadratics_improvement"] = quad_improvement
        file["baseline_kl_xy"] = baseline_metrics.kl_xy
        file["model_kl_xy"] = model_metrics.kl_xy
        for result in observable_results
            group = create_group(file, slugify(result.name))
            group["C_obs"] = result.C_obs
            group["C_model"] = result.C_model
            group["C_baseline"] = result.baseline.C_model
            group["relative_error"] = result.relative_error
            group["baseline_relative_error"] = result.baseline.relative_error
            group["relative_rmse"] = result.relative_rmse
            group["baseline_relative_rmse"] = result.baseline.relative_rmse
            for (label_idx, label) in enumerate(result.output_labels)
                group["output_label_$(label_idx)"] = label
            end
        end
    end

    return Dict(
        "status" => "completed",
        "delta_m_rollout_h5" => paths.delta_m_rollout_h5,
        "delta_m_rollout_png" => paths.delta_m_rollout_png,
        "delta_m_rollout_pdf" => paths.delta_m_rollout_pdf,
        "coordinates_relative_rmse" => coord_result.relative_rmse,
        "coordinates_baseline_relative_rmse" => coord_baseline_rmse,
        "quadratics_relative_rmse" => quad_result.relative_rmse,
        "quadratics_baseline_relative_rmse" => quad_baseline_rmse,
        "coordinates_improvement" => coord_improvement,
        "quadratics_improvement" => quad_improvement,
    )
end

function run_delta_m_coefficients(config::Dict{String,Any}, paths; force::Bool=false)
    if !should_run_stage(paths, :delta_m_coefficients, force)
        return Dict("status" => "skipped", "delta_m_coefficients_png" => paths.delta_m_coefficients_png)
    end

    require_artifact(paths.dataset_h5, :generate_data, :delta_m_coefficients)
    require_artifact(paths.score_model_bson, :train_score, :delta_m_coefficients)
    require_artifact(paths.phi_sigma_h5, :estimate_phi_sigma, :delta_m_coefficients)
    require_artifact(paths.delta_m_model_bson, :delta_m, :delta_m_coefficients)
    require_artifact(paths.delta_m_rollout_h5, :delta_m_rollout, :delta_m_coefficients)

    data = load_dataset(paths.dataset_h5)
    coeff_cfg = config["delta_m_coefficients"]
    score_wrapper, _ = build_score_wrapper(paths.score_model_bson)
    delta_model, delta_meta = load_model(paths.delta_m_model_bson)
    divergence_mode = resolve_divergence_mode(get(delta_meta, "divergence_mode", get(config["delta_m"], "divergence_mode", "autodiff")))
    phi_sigma = h5open(paths.phi_sigma_h5, "r") do file
        (
            Phi=Float32.(read(file["Phi"])),
            Sigma=Float32.(read(file["Sigma"])),
        )
    end
    delta_metrics = h5open(paths.delta_m_h5, "r") do file
        Dict(
            "selected_candidate_name" => read(file["selected_candidate_name"]),
            "coordinates_residual_rmse" => Float64(read(file["coordinates/relative_rmse"])),
            "quadratics_residual_rmse" => Float64(read(file["centered_quadratics/relative_rmse"])),
            "mean_delta_norm" => Float64(read(file["mean_delta_norm"])),
            "min_lambda_samples" => Float64(read(file["min_lambda_samples"])),
        )
    end
    rollout_metrics = h5open(paths.delta_m_rollout_h5, "r") do file
        Dict(
            "coordinates_relative_rmse" => Float64(read(file["coordinates_relative_rmse"])),
            "coordinates_baseline_relative_rmse" => Float64(read(file["coordinates_baseline_relative_rmse"])),
            "quadratics_relative_rmse" => Float64(read(file["quadratics_relative_rmse"])),
            "quadratics_baseline_relative_rmse" => Float64(read(file["quadratics_baseline_relative_rmse"])),
            "coordinates_improvement" => Float64(read(file["coordinates_improvement"])),
            "quadratics_improvement" => Float64(read(file["quadratics_improvement"])),
        )
    end

    axes_info = percentile_field_axes(
        data.uncorrelated,
        Int(coeff_cfg["grid_nx"]),
        Int(coeff_cfg["grid_ny"]);
        lower_quantile=Float64(coeff_cfg["lower_quantile"]),
        upper_quantile=Float64(coeff_cfg["upper_quantile"]),
        padding_fraction=Float64(coeff_cfg["padding_fraction"]),
    )
    xgrid = axes_info.x
    ygrid = axes_info.y
    density, weights = stationary_density_grid(data.uncorrelated, xgrid, ygrid)
    true_cfg = build_generation_config(config)
    truth_drift, truth_diffusion = true_brusselator_fields(true_cfg, xgrid, ygrid)
    base_drift, base_diffusion = baseline_rom_fields(score_wrapper, phi_sigma.Phi, phi_sigma.Sigma, xgrid, ygrid)
    grid_data = build_delta_m_grids(delta_model, score_wrapper, axes_info; batch_size=8192, use_gpu=false, divergence_mode=divergence_mode)
    model_drift = grid_data.drift_grid
    model_score_term = grid_data.score_term_grid
    model_divergence = grid_data.divergence_grid
    model_diffusion = sigma_grid_to_diffusion_tensor(grid_data.sigma_grid)
    affine_lag_steps = resolve_delta_m_rollout_lag_steps(
        data,
        Int.(delta_meta["lag_steps"]),
        config["delta_m_rollout"],
        Int(get(config["delta_m"], "lag_step_stride", 1)),
    )
    affine_lag_times = Float64.(affine_lag_steps) .* data.sample_dt
    affine_metrics = coordinate_affine_closure_metrics(data.correlated, affine_lag_steps)
    metrics, drift_base_norm, drift_model_norm, diff_base_norm, diff_model_norm = coefficient_metrics(
        truth_drift,
        truth_diffusion,
        base_drift,
        base_diffusion,
        model_drift,
        model_diffusion,
        weights,
    )

    drift_weighted_baseline = weighted_rmse(drift_base_norm, weights)
    drift_weighted_model = weighted_rmse(drift_model_norm, weights)
    drift_full_baseline = sqrt(mean(abs2, drift_base_norm))
    drift_full_model = sqrt(mean(abs2, drift_model_norm))
    diff_weighted_baseline = weighted_rmse(diff_base_norm, weights)
    diff_weighted_model = weighted_rmse(diff_model_norm, weights)
    diff_full_baseline = sqrt(mean(abs2, diff_base_norm))
    diff_full_model = sqrt(mean(abs2, diff_model_norm))
    drift_verdict = drift_weighted_model < drift_weighted_baseline ? "improved" : (drift_weighted_model > drift_weighted_baseline ? "worsened" : "unchanged")
    diff_verdict = diff_weighted_model < diff_weighted_baseline ? "improved" : (diff_weighted_model > diff_weighted_baseline ? "worsened" : "unchanged")
    overall_verdict = drift_verdict == diff_verdict ? drift_verdict : "mixed"

    summary_lines = [
        "Analytical coefficients are evaluation-only.",
        "selected δM candidate = $(delta_metrics["selected_candidate_name"])",
        "divergence mode = $(String(divergence_mode))",
        "Residual RMSE coords = $(round(delta_metrics["coordinates_residual_rmse"], digits=5))",
        "Residual RMSE quads = $(round(delta_metrics["quadratics_residual_rmse"], digits=5))",
        "Rollout RMSE coords: baseline $(round(rollout_metrics["coordinates_baseline_relative_rmse"], digits=5)) | δM $(round(rollout_metrics["coordinates_relative_rmse"], digits=5))",
        "Rollout RMSE quads: baseline $(round(rollout_metrics["quadratics_baseline_relative_rmse"], digits=5)) | δM $(round(rollout_metrics["quadratics_relative_rmse"], digits=5))",
        "Drift weighted RMSE: baseline $(round(drift_weighted_baseline, digits=5)) | δM $(round(drift_weighted_model, digits=5))",
        "Diffusion weighted RMSE: baseline $(round(diff_weighted_baseline, digits=5)) | δM $(round(diff_weighted_model, digits=5))",
        "Drift full-grid RMSE: baseline $(round(drift_full_baseline, digits=5)) | δM $(round(drift_full_model, digits=5))",
        "Diffusion full-grid RMSE: baseline $(round(diff_full_baseline, digits=5)) | δM $(round(diff_full_model, digits=5))",
        "Coefficient verdict: drift $(drift_verdict), diffusion $(diff_verdict), overall $(overall_verdict)",
        "mean ||div(M)|| = $(round(mean(sqrt.(sum(abs2, Float64.(model_divergence); dims=1))[1, :, :]), digits=5))",
        "mean affine-closure rel. RMSE = $(round(mean(affine_metrics.rel_rmse), digits=5))",
        "mean ||δM|| diagnostic = $(round(delta_metrics["mean_delta_norm"], digits=5))",
        "min λ(Dθ) on samples = $(round(delta_metrics["min_lambda_samples"], digits=5))",
    ]

    fig = delta_m_coefficients_figure(
        xgrid,
        ygrid,
        density,
        truth_drift,
        truth_diffusion,
        base_drift,
        base_diffusion,
        model_drift,
        model_diffusion,
        metrics,
        drift_base_norm,
        drift_model_norm,
        diff_base_norm,
        diff_model_norm,
        summary_lines,
    )
    save(paths.delta_m_coefficients_png, fig, px_per_unit=2)
    save(paths.delta_m_coefficients_pdf, fig)

    h5open(paths.delta_m_coefficients_h5, "w") do file
        file["xgrid"] = Float64.(xgrid)
        file["ygrid"] = Float64.(ygrid)
        file["density"] = density
        file["weights"] = weights
        file["truth_drift"] = truth_drift
        file["truth_diffusion"] = truth_diffusion
        file["baseline_drift"] = base_drift
        file["baseline_diffusion"] = base_diffusion
        file["model_drift"] = model_drift
        file["model_score_term"] = model_score_term
        file["model_divergence"] = model_divergence
        file["model_diffusion"] = model_diffusion
        file["baseline_drift_abs_error"] = abs.(Float32.(base_drift .- truth_drift))
        file["model_drift_abs_error"] = abs.(Float32.(model_drift .- truth_drift))
        file["model_score_term_abs_error_vs_truth_drift"] = abs.(Float32.(model_score_term .- truth_drift))
        file["model_divergence_abs_value"] = abs.(Float32.(model_divergence))
        file["baseline_diffusion_abs_error"] = abs.(Float32.(base_diffusion .- truth_diffusion))
        file["model_diffusion_abs_error"] = abs.(Float32.(model_diffusion .- truth_diffusion))
        file["drift_norm_error_baseline"] = Float32.(drift_base_norm)
        file["drift_norm_error_model"] = Float32.(drift_model_norm)
        file["diffusion_norm_error_baseline"] = Float32.(diff_base_norm)
        file["diffusion_norm_error_model"] = Float32.(diff_model_norm)
        file["drift_weighted_rmse_baseline"] = drift_weighted_baseline
        file["drift_weighted_rmse_model"] = drift_weighted_model
        file["drift_full_rmse_baseline"] = drift_full_baseline
        file["drift_full_rmse_model"] = drift_full_model
        file["diffusion_weighted_rmse_baseline"] = diff_weighted_baseline
        file["diffusion_weighted_rmse_model"] = diff_weighted_model
        file["diffusion_full_rmse_baseline"] = diff_full_baseline
        file["diffusion_full_rmse_model"] = diff_full_model
        file["selected_candidate_name"] = delta_metrics["selected_candidate_name"]
        file["coordinates_residual_rmse"] = delta_metrics["coordinates_residual_rmse"]
        file["quadratics_residual_rmse"] = delta_metrics["quadratics_residual_rmse"]
        file["divergence_mode"] = String(divergence_mode)
        file["coordinate_affine_lag_steps"] = Int.(affine_lag_steps)
        file["coordinate_affine_lag_times"] = Float64.(affine_lag_times)
        file["coordinate_affine_rel_rmse"] = Float64.(affine_metrics.rel_rmse)
        file["coordinate_affine_A"] = Float64.(affine_metrics.A)
        file["coordinates_rollout_relative_rmse"] = rollout_metrics["coordinates_relative_rmse"]
        file["coordinates_rollout_baseline_relative_rmse"] = rollout_metrics["coordinates_baseline_relative_rmse"]
        file["quadratics_rollout_relative_rmse"] = rollout_metrics["quadratics_relative_rmse"]
        file["quadratics_rollout_baseline_relative_rmse"] = rollout_metrics["quadratics_baseline_relative_rmse"]
        file["coordinates_rollout_improvement"] = rollout_metrics["coordinates_improvement"]
        file["quadratics_rollout_improvement"] = rollout_metrics["quadratics_improvement"]
        metrics_group = create_group(file, "component_metrics")
        for name in ("b1", "b2", "G11", "G12", "G22")
            group = create_group(metrics_group, name)
            vals = metrics[name]
            group["full_baseline_rmse"] = vals.full_baseline_rmse
            group["full_model_rmse"] = vals.full_model_rmse
            group["weighted_baseline_rmse"] = vals.weighted_baseline_rmse
            group["weighted_model_rmse"] = vals.weighted_model_rmse
        end
    end

    return Dict(
        "status" => "completed",
        "delta_m_coefficients_h5" => paths.delta_m_coefficients_h5,
        "delta_m_coefficients_png" => paths.delta_m_coefficients_png,
        "delta_m_coefficients_pdf" => paths.delta_m_coefficients_pdf,
        "drift_weighted_rmse_baseline" => drift_weighted_baseline,
        "drift_weighted_rmse_model" => drift_weighted_model,
        "diffusion_weighted_rmse_baseline" => diff_weighted_baseline,
        "diffusion_weighted_rmse_model" => diff_weighted_model,
        "coordinate_affine_rel_rmse_mean" => mean(affine_metrics.rel_rmse),
        "coefficient_verdict" => overall_verdict,
    )
end

function run_stage(stage::Symbol, config::Dict{String,Any}, paths; force::Bool=false)
    stage_fn = if stage === :generate_data
        () -> run_generate_data(config, paths; force=force)
    elseif stage === :train_score
        () -> run_train_score(config, paths; force=force)
    elseif stage === :estimate_phi_sigma
        () -> run_estimate_phi_sigma(config, paths; force=force)
    elseif stage === :observed_residuals
        () -> run_observed_residuals(config, paths; force=force)
    elseif stage === :overview
        () -> run_overview(config, paths; force=force)
    elseif stage === :joint_score
        () -> run_joint_score(config, paths; force=force)
    elseif stage === :delta_m
        () -> run_delta_m(config, paths; force=force)
    elseif stage === :delta_m_rollout
        () -> run_delta_m_rollout(config, paths; force=force)
    elseif stage === :delta_m_coefficients
        () -> run_delta_m_coefficients(config, paths; force=force)
    else
        error("Unsupported stage `$(stage)`.")
    end

    result, elapsed = stage_elapsed(stage_fn)
    summary_info = Dict{String,Any}(result)
    summary_info["elapsed_seconds"] = elapsed
    update_stage_summary!(paths, stage, summary_info)
    return result
end

function load_config(args::Dict{String,String})
    return validate_config(TOML.parsefile(config_path(args)))
end

function run_cli(default_stage::Union{Nothing,Symbol}=nothing)
    args = parse_args()
    config = load_config(args)
    paths = artifact_paths(config, args)
    initialize_run!(config, paths, config_path(args))

    force = parse_bool(get(args, "force", get(config["run"], "force", false)))
    stages = requested_stages(config, args, default_stage)

    println("Brusselator run directory: $(paths.run_dir)")
    println("Requested stages: $(join(String.(stages), ", "))")

    for stage in stages
        result = run_stage(stage, config, paths; force=force)
        println("[$(stage)] $(result)")
    end

    return nothing
end

end # module
