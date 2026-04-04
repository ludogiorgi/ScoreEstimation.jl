Base.@kwdef struct ExecutionConfig
    stages::Vector{Symbol}
end

Base.@kwdef struct SystemConfig
    label::String
    forcing_x1::Symbol
    forcing_x2::Symbol
    eps::Float64
    kappa::Float64
    Omega::Float64
    a1::Float64
    a2::Float64
    b1::Float64
    b2::Float64
    sigma_x1::Float64
    sigma_x2::Float64
    base_y2_ref::Float64
    base_y3_ref::Float64
end

Base.@kwdef struct HighResolutionConfig
    dt::Float64
    t_reference::Float64
    t_reference_transient::Float64
    t_total::Float64
    t_transient::Float64
    sample_stride::Int
end

Base.@kwdef struct LowResolutionConfig
    training_target_uncorrelated::Int
    training_seed_stride_multiplier::Int
end

Base.@kwdef struct ScoreConfig
    rng_seed::Int
    noise_scales::Vector{Float64}
    epochs::Int
    batch_size::Int
    neurons::Vector{Int}
    learning_rate::Float64
    use_gpu::Bool
    rollout_device::Symbol
    mean_regularization::Float64
    stein_regularization::Float64
    max_training_samples::Int
    max_stationary_samples::Int
    max_batches_per_epoch::Union{Nothing,Int}
end

Base.@kwdef struct PhiSigmaConfig
    minimum_probability::Float64
    partition_override::Bool
    generator_perturb_scale::Float64
    stationary_perturb_scale::Float64
    sigma_regularization::Float64
    use_identity_phi::Bool
    use_identity_sigma::Bool
    max_correlated_samples::Int
end

Base.@kwdef struct RomConfig
    maxlag::Int
    phase_points::Int
    pdf_bins::Int
    joint_pdf_bins::Int
    candidate_rollout_dt::Float64
    candidate_rollout_steps::Int
    candidate_rollout_burnin::Int
    candidate_rollout_ensembles::Int
    final_rollout_dt::Float64
    final_rollout_steps::Int
    final_rollout_burnin::Int
    final_rollout_ensembles::Int
    x_excerpt_points::Int
    score_grid_n::Int
end

Base.@kwdef struct ResidualConfig
    derivative_window_radius::Int
    derivative_degree::Int
    excerpt_points::Int
    pdf_bins::Int
    acf_lags::Int
end

Base.@kwdef struct CoupledL63Config
    execution::ExecutionConfig
    system::SystemConfig
    high_resolution::HighResolutionConfig
    low_resolution::LowResolutionConfig
    score::ScoreConfig
    phi_sigma::PhiSigmaConfig
    rom::RomConfig
    residuals::ResidualConfig
    raw::Dict{String,Any}
end

parse_bool(value::Bool) = value
parse_bool(value) = lowercase(strip(string(value))) in ("1", "true", "yes", "on")

function _optional_int(cfg::Dict{String,Any}, key::String)
    haskey(cfg, key) || return nothing
    return Int(cfg[key])
end

canonical_stage(stage::Symbol) = stage
function canonical_stage(stage)
    value = Symbol(lowercase(strip(String(stage))))
    value in STAGE_SEQUENCE || error("Unsupported stage `$(stage)`.")
    return value
end

function require_section(config::Dict{String,Any}, key::String)
    haskey(config, key) || error("Missing required config section `$(key)` in $(DEFAULT_CONFIG).")
    return config[key]
end

function parse_config(config::Dict{String,Any})
    execution_cfg = require_section(config, "execution")
    system_cfg = require_section(config, "system")
    integration_cfg = require_section(config, "integration")
    high_cfg = require_section(integration_cfg, "high_resolution")
    low_cfg = require_section(integration_cfg, "low_resolution")
    score_cfg = require_section(config, "score")
    phi_cfg = require_section(config, "phi_sigma")
    rom_cfg = require_section(config, "rom")
    residual_cfg = require_section(config, "residuals")

    execution = ExecutionConfig(
        stages=[canonical_stage(stage) for stage in get(execution_cfg, "stages", String["generate_data", "fit_score", "fit_phi_sigma", "estimate_residuals"])],
    )

    system = SystemConfig(
        label=String(get(system_cfg, "label", "Final coupled L63 system")),
        forcing_x1=Symbol(String(system_cfg["forcing_x1"])),
        forcing_x2=Symbol(String(system_cfg["forcing_x2"])),
        eps=Float64(system_cfg["eps"]),
        kappa=Float64(system_cfg["kappa"]),
        Omega=Float64(system_cfg["Omega"]),
        a1=Float64(system_cfg["a1"]),
        a2=Float64(system_cfg["a2"]),
        b1=Float64(system_cfg["b1"]),
        b2=Float64(system_cfg["b2"]),
        sigma_x1=Float64(system_cfg["sigma_x1"]),
        sigma_x2=Float64(system_cfg["sigma_x2"]),
        base_y2_ref=Float64(get(system_cfg, "base_y2_ref", 0.0)),
        base_y3_ref=Float64(get(system_cfg, "base_y3_ref", 0.0)),
    )

    high_resolution = HighResolutionConfig(
        dt=Float64(high_cfg["dt"]),
        t_reference=Float64(high_cfg["t_reference"]),
        t_reference_transient=Float64(high_cfg["t_reference_transient"]),
        t_total=Float64(high_cfg["t_total"]),
        t_transient=Float64(high_cfg["t_transient"]),
        sample_stride=Int(high_cfg["sample_stride"]),
    )

    low_resolution = LowResolutionConfig(
        training_target_uncorrelated=Int(get(low_cfg, "training_target_uncorrelated", 100_000)),
        training_seed_stride_multiplier=Int(get(low_cfg, "training_seed_stride_multiplier", 1)),
    )

    score = ScoreConfig(
        rng_seed=Int(get(score_cfg, "rng_seed", 20260402)),
        noise_scales=Float64.(get(score_cfg, "noise_scales", [0.02, 0.03, 0.04])),
        epochs=Int(get(score_cfg, "epochs", 320)),
        batch_size=Int(get(score_cfg, "batch_size", 2048)),
        neurons=Int.(get(score_cfg, "neurons", [192, 192, 128])),
        learning_rate=Float64(get(score_cfg, "learning_rate", 6.0e-4)),
        use_gpu=parse_bool(get(score_cfg, "use_gpu", true)),
        rollout_device=Symbol(get(score_cfg, "rollout_device", "auto")),
        mean_regularization=Float64(get(score_cfg, "mean_regularization", 1.0e-3)),
        stein_regularization=Float64(get(score_cfg, "stein_regularization", 6.0e-2)),
        max_training_samples=Int(get(score_cfg, "max_training_samples", 180_000)),
        max_stationary_samples=Int(get(score_cfg, "max_stationary_samples", 180_000)),
        max_batches_per_epoch=_optional_int(score_cfg, "max_batches_per_epoch"),
    )

    phi_sigma = PhiSigmaConfig(
        minimum_probability=Float64(get(phi_cfg, "minimum_probability", 2.0e-2)),
        partition_override=parse_bool(get(phi_cfg, "partition_override", true)),
        generator_perturb_scale=Float64(get(phi_cfg, "generator_perturb_scale", -1.0)),
        stationary_perturb_scale=Float64(get(phi_cfg, "stationary_perturb_scale", -1.0)),
        sigma_regularization=Float64(get(phi_cfg, "sigma_regularization", 8.0e-4)),
        use_identity_phi=parse_bool(get(phi_cfg, "use_identity_phi", false)),
        use_identity_sigma=parse_bool(get(phi_cfg, "use_identity_sigma", false)),
        max_correlated_samples=Int(get(phi_cfg, "max_correlated_samples", 240_000)),
    )

    rom = RomConfig(
        maxlag=Int(get(rom_cfg, "maxlag", 1800)),
        phase_points=Int(get(rom_cfg, "phase_points", 18_000)),
        pdf_bins=Int(get(rom_cfg, "pdf_bins", 140)),
        joint_pdf_bins=Int(get(rom_cfg, "joint_pdf_bins", 100)),
        candidate_rollout_dt=Float64(get(rom_cfg, "candidate_rollout_dt", 0.005)),
        candidate_rollout_steps=Int(get(rom_cfg, "candidate_rollout_steps", 70_000)),
        candidate_rollout_burnin=Int(get(rom_cfg, "candidate_rollout_burnin", 16_000)),
        candidate_rollout_ensembles=Int(get(rom_cfg, "candidate_rollout_ensembles", 128)),
        final_rollout_dt=Float64(get(rom_cfg, "final_rollout_dt", 0.005)),
        final_rollout_steps=Int(get(rom_cfg, "final_rollout_steps", 200_000)),
        final_rollout_burnin=Int(get(rom_cfg, "final_rollout_burnin", 26_000)),
        final_rollout_ensembles=Int(get(rom_cfg, "final_rollout_ensembles", 256)),
        x_excerpt_points=Int(get(rom_cfg, "x_excerpt_points", 3000)),
        score_grid_n=Int(get(rom_cfg, "score_grid_n", 64)),
    )

    residuals = ResidualConfig(
        derivative_window_radius=Int(get(residual_cfg, "derivative_window_radius", 7)),
        derivative_degree=Int(get(residual_cfg, "derivative_degree", 3)),
        excerpt_points=Int(get(residual_cfg, "excerpt_points", 1000)),
        pdf_bins=Int(get(residual_cfg, "pdf_bins", 180)),
        acf_lags=Int(get(residual_cfg, "acf_lags", 1000)),
    )

    return CoupledL63Config(
        execution=execution,
        system=system,
        high_resolution=high_resolution,
        low_resolution=low_resolution,
        score=score,
        phi_sigma=phi_sigma,
        rom=rom,
        residuals=residuals,
        raw=config,
    )
end
