"""
    save_model(model, filename; metadata=nothing)

Persist a Flux model to disk using BSON. The stored model is moved to the CPU
before serialization so the resulting artifact can be reloaded independently of
the training device.
"""
function save_model(model, filename; metadata=nothing)
    path = endswith(filename, ".bson") ? filename : string(filename, ".bson")
    mkpath(dirname(path))
    model_cpu = Flux.cpu(model)
    BSON.@save path model_cpu metadata
    return path
end

"""
    load_model(filename) -> (model, metadata)

Reload a model saved with [`save_model`](@ref). Returns the model and any
optional metadata dictionary that was persisted alongside it.
"""
function load_model(filename)
    path = endswith(filename, ".bson") ? filename : string(filename, ".bson")
    data = BSON.load(path)
    model = haskey(data, :model_cpu) ? data[:model_cpu] : data[:model]
    metadata = get(data, :metadata, nothing)
    return model, metadata
end

"""
    save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")

Save a dictionary of variables to an HDF5 file.
"""
function save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")
    mkpath(dirname(filename))
    h5open(filename, "w") do file
        group = group_path == "/" ? file : create_group(file, group_path)
        for (name, value) in vars
            if value isa AbstractArray
                group[name] = value
            elseif value isa Number || value isa Bool
                dataset = group[name] = [value]
                attrs(dataset)["scalar"] = true
            elseif value isa String
                group[name] = value
            elseif value isa Symbol
                dataset = group[name] = string(value)
                attrs(dataset)["symbol"] = true
            else
                @warn "Skipping unsupported value while writing HDF5" name type=typeof(value)
            end
        end
    end
    return filename
end

"""
    read_variables_from_hdf5(filename::String; group_path="/")

Read variables from an HDF5 file into a dictionary.
"""
function read_variables_from_hdf5(filename::String; group_path="/")
    vars = Dict{String,Any}()
    h5open(filename, "r") do file
        group = group_path == "/" ? file : file[group_path]
        for name in keys(group)
            dataset = group[name]
            value = read(dataset)
            if haskey(attrs(dataset), "scalar") && attrs(dataset)["scalar"]
                vars[String(name)] = value[1]
            elseif haskey(attrs(dataset), "symbol") && attrs(dataset)["symbol"]
                vars[String(name)] = Symbol(value)
            else
                vars[String(name)] = value
            end
        end
    end
    return vars
end

"""
    load_to_workspace(filename::String; overwrite=false)

Load variables from an HDF5 file into `Main`.
"""
function load_to_workspace(filename::String; overwrite=false)
    vars = read_variables_from_hdf5(filename)
    for (name, value) in vars
        sym = Symbol(name)
        if !overwrite && isdefined(Main, sym)
            @warn "Skipping variable already defined in Main" variable=name
            continue
        end
        Core.eval(Main, :($(sym) = $(value)))
    end
    return nothing
end
