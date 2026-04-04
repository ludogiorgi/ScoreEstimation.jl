Base.@kwdef struct ModelParameters
    eps::Float64
    kappa::Float64
    Omega::Float64
    sigma_x1::Float64
    sigma_x2::Float64
    forcing_x1::Symbol
    forcing_x2::Symbol
    a1::Float64
    a2::Float64
    b1::Float64
    b2::Float64
    y2_ref::Float64
    y3_ref::Float64
end

Base.@kwdef struct ReferenceLevels
    y2_ref::Float64
    y3_ref::Float64
    std_y2::Float64
    std_y3::Float64
end

Base.@kwdef struct SimulationData
    t::Vector{Float64}
    x1::Vector{Float64}
    x2::Vector{Float64}
    y1::Vector{Float64}
    y2::Vector{Float64}
    y3::Vector{Float64}
    dx1::Vector{Float64}
    dx2::Vector{Float64}
    sample_dt::Float64
    maxabs::Float64
end

mutable struct RK4Workspace
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    tmp::Vector{Float64}
end

RK4Workspace(n::Int) = RK4Workspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))

observed_matrix(data::SimulationData) = Matrix{Float64}(vcat(permutedims(data.x1), permutedims(data.x2)))

function state_matrix(data::SimulationData)
    return Matrix{Float64}(vcat(
        permutedims(data.x1),
        permutedims(data.x2),
        permutedims(data.y1),
        permutedims(data.y2),
        permutedims(data.y3),
    ))
end

@inline inv_eps2(p::ModelParameters) = 1.0 / (p.eps * p.eps)

function full_rhs!(du::Vector{Float64}, u::Vector{Float64}, p::ModelParameters)
    x1, x2, y1, y2, y3 = u
    ieps2 = inv_eps2(p)
    forcing1 = p.forcing_x1 === :y2 ? (y2 - p.y2_ref) : (y3 - p.y3_ref)
    forcing2 = p.forcing_x2 === :y2 ? (y2 - p.y2_ref) : (y3 - p.y3_ref)

    du[1] = x1 - x1^3 + p.Omega * x2 + p.sigma_x1 * forcing1
    du[2] = -p.kappa * x2 - p.Omega * x1 + p.sigma_x2 * forcing2
    du[3] = ieps2 * (10.0 * (y2 - y1) + p.a1 * x1 + p.a2 * x2)
    du[4] = ieps2 * (((28.0 + p.b1 * x1 + p.b2 * x2) * y1) - y2 - y1 * y3)
    du[5] = ieps2 * (y1 * y2 - (8.0 / 3.0) * y3)
    return nothing
end

function reference_fast_rhs!(du::Vector{Float64}, y::Vector{Float64}, p::ModelParameters)
    y1, y2, y3 = y
    ieps2 = inv_eps2(p)
    du[1] = ieps2 * (10.0 * (y2 - y1))
    du[2] = ieps2 * (28.0 * y1 - y2 - y1 * y3)
    du[3] = ieps2 * (y1 * y2 - (8.0 / 3.0) * y3)
    return nothing
end

function rk4_step!(u::Vector{Float64}, dt::Float64, rhs!, p, work::RK4Workspace)
    rhs!(work.k1, u, p)
    @inbounds for j in eachindex(u)
        work.tmp[j] = u[j] + 0.5 * dt * work.k1[j]
    end
    rhs!(work.k2, work.tmp, p)
    @inbounds for j in eachindex(u)
        work.tmp[j] = u[j] + 0.5 * dt * work.k2[j]
    end
    rhs!(work.k3, work.tmp, p)
    @inbounds for j in eachindex(u)
        work.tmp[j] = u[j] + dt * work.k3[j]
    end
    rhs!(work.k4, work.tmp, p)
    @inbounds for j in eachindex(u)
        u[j] += (dt / 6.0) * (work.k1[j] + 2.0 * work.k2[j] + 2.0 * work.k3[j] + work.k4[j])
    end
    return nothing
end

sample_count(nsteps::Int, transient_steps::Int, stride::Int) = fld(nsteps, stride) - fld(transient_steps, stride)

function base_parameters(system::SystemConfig)
    return ModelParameters(
        eps=system.eps,
        kappa=system.kappa,
        Omega=system.Omega,
        sigma_x1=0.0,
        sigma_x2=0.0,
        forcing_x1=system.forcing_x1,
        forcing_x2=system.forcing_x2,
        a1=system.a1,
        a2=system.a2,
        b1=system.b1,
        b2=system.b2,
        y2_ref=system.base_y2_ref,
        y3_ref=system.base_y3_ref,
    )
end

function build_parameters(system::SystemConfig, refs::ReferenceLevels)
    return ModelParameters(
        eps=system.eps,
        kappa=system.kappa,
        Omega=system.Omega,
        sigma_x1=system.sigma_x1,
        sigma_x2=system.sigma_x2,
        forcing_x1=system.forcing_x1,
        forcing_x2=system.forcing_x2,
        a1=system.a1,
        a2=system.a2,
        b1=system.b1,
        b2=system.b2,
        y2_ref=refs.y2_ref,
        y3_ref=refs.y3_ref,
    )
end

function estimate_reference_levels(system::SystemConfig, cfg::HighResolutionConfig)
    base = base_parameters(system)
    y = [1.0, 1.0, 1.0]
    work = RK4Workspace(3)
    nsteps = round(Int, cfg.t_reference / cfg.dt)
    transient_steps = round(Int, cfg.t_reference_transient / cfg.dt)
    nsamples = sample_count(nsteps, transient_steps, cfg.sample_stride)
    report_stride = max(cld(nsteps, 5), 1)
    log_message(
        "generate_data",
        @sprintf(
            "Reference-level integration started: nsteps=%d, transient_steps=%d, sample_stride=%d, samples=%d",
            nsteps,
            transient_steps,
            cfg.sample_stride,
            nsamples,
        ),
    )
    samples_y2 = Vector{Float64}(undef, nsamples)
    samples_y3 = similar(samples_y2)
    idx = 0
    for step in 1:nsteps
        rk4_step!(y, cfg.dt, reference_fast_rhs!, base, work)
        if step > transient_steps && step % cfg.sample_stride == 0
            idx += 1
            samples_y2[idx] = y[2]
            samples_y3[idx] = y[3]
        end
        if step == nsteps || step % report_stride == 0
            log_message(
                "generate_data",
                @sprintf("Reference-level integration progress: %d/%d steps (%.1f%%)", step, nsteps, 100 * step / nsteps),
            )
        end
    end
    refs = ReferenceLevels(
        y2_ref=mean(samples_y2) + system.base_y2_ref,
        y3_ref=mean(samples_y3) + system.base_y3_ref,
        std_y2=std(samples_y2),
        std_y3=std(samples_y3),
    )
    log_message(
        "generate_data",
        @sprintf("Reference-level integration finished: std_y2=%.6f, std_y3=%.6f", refs.std_y2, refs.std_y3),
    )
    return refs
end

function simulate_high_resolution(system::SystemConfig, params::ModelParameters, cfg::HighResolutionConfig)
    nsteps = round(Int, cfg.t_total / cfg.dt)
    transient_steps = round(Int, cfg.t_transient / cfg.dt)
    nsamples = sample_count(nsteps, transient_steps, cfg.sample_stride)
    report_stride = max(cld(nsteps, 10), 1)
    log_message(
        "generate_data",
        @sprintf(
            "High-resolution simulation started: nsteps=%d, transient_steps=%d, sample_stride=%d, expected_samples=%d",
            nsteps,
            transient_steps,
            cfg.sample_stride,
            nsamples,
        ),
    )
    u = [1.0, 0.0, 1.0, 1.0, 1.0]
    work = RK4Workspace(5)
    du = zeros(5)
    t = Vector{Float64}(undef, nsamples)
    x1 = Vector{Float64}(undef, nsamples)
    x2 = Vector{Float64}(undef, nsamples)
    y1 = Vector{Float64}(undef, nsamples)
    y2 = Vector{Float64}(undef, nsamples)
    y3 = Vector{Float64}(undef, nsamples)
    dx1 = Vector{Float64}(undef, nsamples)
    dx2 = Vector{Float64}(undef, nsamples)
    idx = 0
    maxabs = 0.0
    for step in 1:nsteps
        rk4_step!(u, cfg.dt, full_rhs!, params, work)
        maxabs = max(maxabs, maximum(abs, u))
        if !all(isfinite, u) || maxabs > 1.0e6
            error(@sprintf("High-resolution simulation became unstable for %s.", system.label))
        end
        if step > transient_steps && step % cfg.sample_stride == 0
            idx += 1
            full_rhs!(du, u, params)
            t[idx] = step * cfg.dt
            x1[idx] = u[1]
            x2[idx] = u[2]
            y1[idx] = u[3]
            y2[idx] = u[4]
            y3[idx] = u[5]
            dx1[idx] = du[1]
            dx2[idx] = du[2]
        end
        if step == nsteps || step % report_stride == 0
            log_message(
                "generate_data",
                @sprintf(
                    "High-resolution simulation progress: %d/%d steps (%.1f%%), saved_samples=%d",
                    step,
                    nsteps,
                    100 * step / nsteps,
                    idx,
                ),
            )
        end
    end
    data = SimulationData(
        t=t,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
        y3=y3,
        dx1=dx1,
        dx2=dx2,
        sample_dt=cfg.dt * cfg.sample_stride,
        maxabs=maxabs,
    )
    log_message("generate_data", @sprintf("High-resolution simulation finished: maxabs=%.6f", maxabs))
    return data
end

function build_low_resolution_dataset(data::SimulationData,
                                      params::ModelParameters,
                                      high_cfg::HighResolutionConfig,
                                      low_cfg::LowResolutionConfig,
                                      system_label::String)
    slow = observed_matrix(data)
    decor_stride_saved, taus = ScoreEstimation.estimate_uncorrelated_stride(slow)
    seed_stride_saved = max(decor_stride_saved * low_cfg.training_seed_stride_multiplier, 1)
    seed_indices = collect(1:seed_stride_saved:length(data.t))
    isempty(seed_indices) && error("No stationary seeds available for low-resolution dataset.")

    seeds = state_matrix(data)[:, seed_indices]
    nseed = size(seeds, 2)
    target = max(low_cfg.training_target_uncorrelated, nseed)
    samples_per_seed = max(1, ceil(Int, target / nseed))
    low_stride_steps = high_cfg.sample_stride * decor_stride_saved
    total = nseed * samples_per_seed
    report_stride = max(cld(nseed, 10), 1)
    completed_seeds = Threads.Atomic{Int}(0)
    log_message(
        "generate_data",
        @sprintf(
            "Low-resolution dataset generation started: seeds=%d, target=%d, samples_per_seed=%d, low_stride_steps=%d, threads=%d",
            nseed,
            target,
            samples_per_seed,
            low_stride_steps,
            nthreads(),
        ),
    )

    x1 = Vector{Float64}(undef, total)
    x2 = similar(x1)
    y1 = similar(x1)
    y2 = similar(x1)
    y3 = similar(x1)
    dx1 = similar(x1)
    dx2 = similar(x1)
    chain_maxabs = zeros(Float64, nseed)

    @threads for seed_id in 1:nseed
        u = Vector{Float64}(seeds[:, seed_id])
        work = RK4Workspace(5)
        du = zeros(5)
        local_maxabs = maximum(abs, u)
        start_idx = (seed_id - 1) * samples_per_seed
        for sample_id in 1:samples_per_seed
            if sample_id > 1
                for _ in 1:low_stride_steps
                    rk4_step!(u, high_cfg.dt, full_rhs!, params, work)
                    local_maxabs = max(local_maxabs, maximum(abs, u))
                    if !all(isfinite, u) || local_maxabs > 1.0e6
                        error(@sprintf("Low-resolution simulation became unstable for %s.", system_label))
                    end
                end
            end

            full_rhs!(du, u, params)
            idx = start_idx + sample_id
            x1[idx] = u[1]
            x2[idx] = u[2]
            y1[idx] = u[3]
            y2[idx] = u[4]
            y3[idx] = u[5]
            dx1[idx] = du[1]
            dx2[idx] = du[2]
        end
        chain_maxabs[seed_id] = local_maxabs
        done_count = Threads.atomic_add!(completed_seeds, 1) + 1
        if done_count == nseed || done_count % report_stride == 0
            log_message(
                "generate_data",
                @sprintf("Low-resolution dataset progress: completed %d/%d seed trajectories", done_count, nseed),
            )
        end
    end

    keep = 1:target
    sample_dt_low = high_cfg.dt * low_stride_steps
    t = collect(0:(target - 1)) .* sample_dt_low
    lowres = SimulationData(
        t=t,
        x1=x1[keep],
        x2=x2[keep],
        y1=y1[keep],
        y2=y2[keep],
        y3=y3[keep],
        dx1=dx1[keep],
        dx2=dx2[keep],
        sample_dt=sample_dt_low,
        maxabs=maximum(chain_maxabs),
    )
    log_message(
        "generate_data",
        @sprintf(
            "Low-resolution dataset finished: produced=%d samples, decor_stride_saved=%d",
            length(lowres.t),
            decor_stride_saved,
        ),
    )
    return lowres, decor_stride_saved, Float64.(taus), nseed, samples_per_seed
end

function save_dataset_artifact(path::String,
                               system::SystemConfig,
                               params::ModelParameters,
                               refs::ReferenceLevels,
                               data::SimulationData;
                               dataset_role::String,
                               metadata::AbstractDict)
    mkpath(dirname(path))
    h5open(path, "w") do file
        file["system_label"] = system.label
        file["created_at"] = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        file["dataset_role"] = dataset_role

        file["time"] = data.t
        file["x1"] = data.x1
        file["x2"] = data.x2
        file["y1"] = data.y1
        file["y2"] = data.y2
        file["y3"] = data.y3
        file["dx1"] = data.dx1
        file["dx2"] = data.dx2

        meta = create_group(file, "metadata")
        meta["sample_dt"] = [data.sample_dt]
        meta["maxabs"] = [data.maxabs]
        for (key, value) in metadata
            if value isa AbstractString
                meta[key] = String(value)
            elseif value isa Bool
                meta[key] = [Int(value)]
            elseif value isa Integer || value isa AbstractFloat
                meta[key] = [value]
            elseif value isa AbstractVector
                meta[key] = collect(value)
            end
        end

        params_group = create_group(file, "parameters")
        params_group["eps"] = [params.eps]
        params_group["kappa"] = [params.kappa]
        params_group["Omega"] = [params.Omega]
        params_group["sigma_x1"] = [params.sigma_x1]
        params_group["sigma_x2"] = [params.sigma_x2]
        params_group["forcing_x1"] = String(params.forcing_x1)
        params_group["forcing_x2"] = String(params.forcing_x2)
        params_group["a1"] = [params.a1]
        params_group["a2"] = [params.a2]
        params_group["b1"] = [params.b1]
        params_group["b2"] = [params.b2]
        params_group["y2_ref"] = [params.y2_ref]
        params_group["y3_ref"] = [params.y3_ref]

        refs_group = create_group(file, "reference_levels")
        refs_group["y2_ref"] = [refs.y2_ref]
        refs_group["y3_ref"] = [refs.y3_ref]
        refs_group["std_y2"] = [refs.std_y2]
        refs_group["std_y3"] = [refs.std_y3]
    end
    return nothing
end

function load_dataset_artifact(path::String)
    h5open(path, "r") do file
        params_group = file["parameters"]
        meta_group = file["metadata"]
        refs_group = file["reference_levels"]
        params = ModelParameters(
            eps=read(params_group["eps"])[1],
            kappa=read(params_group["kappa"])[1],
            Omega=read(params_group["Omega"])[1],
            sigma_x1=read(params_group["sigma_x1"])[1],
            sigma_x2=read(params_group["sigma_x2"])[1],
            forcing_x1=Symbol(String(read(params_group["forcing_x1"]))),
            forcing_x2=Symbol(String(read(params_group["forcing_x2"]))),
            a1=read(params_group["a1"])[1],
            a2=read(params_group["a2"])[1],
            b1=read(params_group["b1"])[1],
            b2=read(params_group["b2"])[1],
            y2_ref=read(params_group["y2_ref"])[1],
            y3_ref=read(params_group["y3_ref"])[1],
        )
        refs = ReferenceLevels(
            y2_ref=read(refs_group["y2_ref"])[1],
            y3_ref=read(refs_group["y3_ref"])[1],
            std_y2=read(refs_group["std_y2"])[1],
            std_y3=read(refs_group["std_y3"])[1],
        )
        data = SimulationData(
            t=read(file["time"]),
            x1=read(file["x1"]),
            x2=read(file["x2"]),
            y1=read(file["y1"]),
            y2=read(file["y2"]),
            y3=read(file["y3"]),
            dx1=read(file["dx1"]),
            dx2=read(file["dx2"]),
            sample_dt=read(meta_group["sample_dt"])[1],
            maxabs=read(meta_group["maxabs"])[1],
        )
        return (
            system_label=String(read(file["system_label"])),
            dataset_role=String(read(file["dataset_role"])),
            params=params,
            refs=refs,
            data=data,
        )
    end
end
