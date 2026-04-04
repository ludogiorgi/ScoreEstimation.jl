Base.@kwdef struct RootPaths
    script_dir::String
    data_dir::String
    runs_dir::String
    renderer_path::String
    parameters_path::String
end

Base.@kwdef struct RunPaths
    run_id::Int
    run_name::String
    run_dir::String
    parameters_snapshot::String
    summary_toml::String
    score_dir::String
    rom_dir::String
    residual_dir::String
    score_plot_data_dir::String
    rom_plot_data_dir::String
    residual_plot_data_dir::String
    score_figure_png::String
    score_figure_pdf::String
    score_training_figure_png::String
    score_training_figure_pdf::String
    rom_figure_png::String
    rom_figure_pdf::String
    residual_figure_png::String
    residual_figure_pdf::String
end

function root_paths(config_path::String)
    return RootPaths(
        script_dir=SCRIPT_DIR,
        data_dir=joinpath(SCRIPT_DIR, "data"),
        runs_dir=joinpath(SCRIPT_DIR, "runs"),
        renderer_path=joinpath(SCRIPT_DIR, "render_coupled_l63_figures.py"),
        parameters_path=config_path,
    )
end

function canonical_string(value)
    if value isa Dict
        keys_sorted = sort!(collect(keys(value)); by=x -> String(x))
        items = String[]
        for key in keys_sorted
            push!(items, string(repr(String(key)), ":", canonical_string(value[key])))
        end
        return "{" * join(items, ",") * "}"
    elseif value isa NamedTuple
        return canonical_string(Dict(string(k) => v for (k, v) in pairs(value)))
    elseif value isa AbstractVector
        return "[" * join(canonical_string.(collect(value)), ",") * "]"
    elseif value isa Tuple
        return "(" * join(canonical_string.(collect(value)), ",") * ")"
    elseif value isa Symbol
        return repr(String(value))
    elseif value isa AbstractString
        return repr(String(value))
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value === nothing
        return "nothing"
    else
        return repr(value)
    end
end

artifact_hash(value) = bytes2hex(SHA.sha1(canonical_string(value)))

function _cache_dir(roots::RootPaths, parts::AbstractVector{<:AbstractString}, hash::AbstractString)
    return joinpath(roots.data_dir, parts..., hash)
end

high_res_cache_dir(roots::RootPaths, hash::AbstractString) = _cache_dir(roots, ["datasets", "high_res"], hash)
low_res_cache_dir(roots::RootPaths, hash::AbstractString) = _cache_dir(roots, ["datasets", "low_res"], hash)
stationary_reference_cache_dir(roots::RootPaths, hash::AbstractString) = _cache_dir(roots, ["stationary_reference"], hash)
score_cache_dir(roots::RootPaths, hash::AbstractString) = _cache_dir(roots, ["scores"], hash)
phi_sigma_cache_dir(roots::RootPaths, hash::AbstractString) = _cache_dir(roots, ["phi_sigma"], hash)

cache_manifest_path(dir::String) = joinpath(dir, "manifest.toml")
cache_parameters_path(dir::String) = joinpath(dir, "parameters.toml")
dataset_h5_path(dir::String) = joinpath(dir, "dataset.h5")
stationary_reference_h5_path(dir::String) = joinpath(dir, "stationary_reference.h5")
score_model_bson_path(dir::String) = joinpath(dir, "score_model.bson")
score_evaluation_h5_path(dir::String) = joinpath(dir, "score_evaluation.h5")
score_summary_path(dir::String) = joinpath(dir, "score_summary.toml")
phi_sigma_h5_path(dir::String) = joinpath(dir, "phi_sigma.h5")
phi_sigma_summary_path(dir::String) = joinpath(dir, "phi_sigma_summary.toml")

function write_toml_file(path::String, value::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, value)
    end
    return nothing
end

function next_run_id(roots::RootPaths)
    mkpath(roots.runs_dir)
    run_ids = Int[]
    for entry in readdir(roots.runs_dir)
        startswith(entry, "run_") || continue
        suffix = entry[5:end]
        all(isdigit, suffix) || continue
        push!(run_ids, parse(Int, suffix))
    end
    return isempty(run_ids) ? 1 : maximum(run_ids) + 1
end

function allocate_run_paths(roots::RootPaths)
    run_id = next_run_id(roots)
    run_name = @sprintf("run_%03d", run_id)
    run_dir = joinpath(roots.runs_dir, run_name)
    score_dir = joinpath(run_dir, "score")
    rom_dir = joinpath(run_dir, "rom")
    residual_dir = joinpath(run_dir, "residuals")
    score_plot_data_dir = joinpath(score_dir, "plot_data")
    rom_plot_data_dir = joinpath(rom_dir, "plot_data")
    residual_plot_data_dir = joinpath(residual_dir, "plot_data")
    for dir in (run_dir, score_dir, rom_dir, residual_dir, score_plot_data_dir, rom_plot_data_dir, residual_plot_data_dir)
        mkpath(dir)
    end
    return RunPaths(
        run_id=run_id,
        run_name=run_name,
        run_dir=run_dir,
        parameters_snapshot=joinpath(run_dir, "parameters.toml"),
        summary_toml=joinpath(run_dir, "run_summary.toml"),
        score_dir=score_dir,
        rom_dir=rom_dir,
        residual_dir=residual_dir,
        score_plot_data_dir=score_plot_data_dir,
        rom_plot_data_dir=rom_plot_data_dir,
        residual_plot_data_dir=residual_plot_data_dir,
        score_figure_png=joinpath(score_dir, "score_validation.png"),
        score_figure_pdf=joinpath(score_dir, "score_validation.pdf"),
        score_training_figure_png=joinpath(score_dir, "score_training_diagnostics.png"),
        score_training_figure_pdf=joinpath(score_dir, "score_training_diagnostics.pdf"),
        rom_figure_png=joinpath(rom_dir, "rom_validation.png"),
        rom_figure_pdf=joinpath(rom_dir, "rom_validation.pdf"),
        residual_figure_png=joinpath(residual_dir, "residual_validation.png"),
        residual_figure_pdf=joinpath(residual_dir, "residual_validation.pdf"),
    )
end

function write_run_parameter_snapshot(run_paths::RunPaths, config::Dict{String,Any})
    write_toml_file(run_paths.parameters_snapshot, config)
    return nothing
end
