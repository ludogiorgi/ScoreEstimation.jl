"""
    save_model(nn, filename)

Save a Flux model and its parameters to disk.

# Arguments
- `nn`: Neural network model
- `filename`: Output filename (will append .bson if not present)
"""
function save_model(nn, filename)
    # Ensure filename has .bson extension
    if !endswith(filename, ".bson")
        filename = filename * ".bson"
    end
    
    # Extract model parameters
    model_params = Flux.params(nn)
    
    # Save model architecture and parameters
    BSON.@save filename model=nn params=model_params
    
    println("Model saved to $filename")
end

"""
    load_model(filename)

Load a Flux model from disk.

# Arguments
- `filename`: Input filename

# Returns
- Loaded neural network model
"""
function load_model(filename)
    # Ensure filename has .bson extension
    if !endswith(filename, ".bson")
        filename = filename * ".bson"
    end
    
    # Load model data
    model_data = BSON.load(filename)
    
    # Extract model and parameters
    model = model_data[:model]
    params = model_data[:params]
    
    # Restore parameters
    Flux.loadparams!(model, params)
    
    println("Model loaded from $filename")
    return model
end

"""
    save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")

Save multiple variables to an HDF5 file with proper type handling.

# Arguments
- `filename`: Path to the HDF5 file to save
- `vars`: Dictionary mapping variable names to their values
- `group_path`: Optional group path within the HDF5 file
"""
function save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    # Save variables to HDF5 file
    h5open(filename, "w") do file
        # Create group if it's not the root
        group = group_path == "/" ? file : create_group(file, group_path)
        
        # Save each variable with type handling
        for (name, value) in vars
            if value isa AbstractArray
                # For arrays, save attributes to track dimensions and type
                dataset = group[name] = value
                if eltype(value) <: Complex
                    # Store complex arrays as tuple of real and imaginary parts
                    group["$(name)_complex"] = true
                    group["$(name)_real"] = real(value)
                    group["$(name)_imag"] = imag(value)
                end
            elseif value isa Number
                # For scalar values
                group[name] = [value]
                attrs(group[name])["scalar"] = true
            elseif value isa String
                # For strings
                group[name] = value
            elseif value isa Bool
                # For booleans
                group[name] = [value]
                attrs(group[name])["boolean"] = true
            elseif value isa Symbol
                # For symbols, convert to string
                group[name] = string(value)
                attrs(group[name])["symbol"] = true
            else
                # Try to convert to array for other types
                try
                    group[name] = [value]
                    attrs(group[name])["custom_type"] = string(typeof(value))
                catch e
                    @warn "Could not save variable $name of type $(typeof(value))"
                end
            end
        end
    end
    
    println("Variables saved to $filename")
end

"""
    save_current_workspace(filename::String; exclude_modules=true, exclude_functions=true)

Save all variables in the current workspace to an HDF5 file.
"""
function save_current_workspace(filename::String; exclude_modules=true, exclude_functions=true)
    # Get all variables in the current workspace
    vars = Dict()
    
    for name in names(Main; all=false)
        # Skip excluded types
        value = getfield(Main, name)
        if (exclude_modules && value isa Module) || 
           (exclude_functions && value isa Function) ||
           name == :ans || name == :exclude_modules || name == :exclude_functions
            continue
        end
        
        # Only save variables that can be serialized
        try
            vars[string(name)] = value
        catch e
            @warn "Skipping variable $name: cannot be serialized"
        end
    end
    
    save_variables_to_hdf5(filename, vars)
end

"""
    read_variables_from_hdf5(filename::String; group_path="/")

Read variables from an HDF5 file, reconstructing their types.

# Returns
- Dictionary of variable names to values
"""
function read_variables_from_hdf5(filename::String; group_path="/")
    vars = Dict()
    
    h5open(filename, "r") do file
        # Access the group
        group = group_path == "/" ? file : file[group_path]
        
        # Read each variable with type handling
        for name in keys(group)
            # Skip special complex array components
            if endswith(name, "_real") || endswith(name, "_imag") || endswith(name, "_complex")
                continue
            end
            
            dataset = group[name]
            
            # Check for complex arrays
            if haskey(group, "$(name)_complex") && group["$(name)_complex"][] == true
                real_part = read(group["$(name)_real"])
                imag_part = read(group["$(name)_imag"])
                vars[name] = real_part + im * imag_part
                continue
            end
            
            # Read the data
            data = read(dataset)
            
            # Handle scalar values
            if haskey(attrs(dataset), "scalar") && attrs(dataset)["scalar"]
                vars[name] = data[1]
            # Handle booleans
            elseif haskey(attrs(dataset), "boolean") && attrs(dataset)["boolean"]
                vars[name] = Bool(data[1])
            # Handle symbols
            elseif haskey(attrs(dataset), "symbol") && attrs(dataset)["symbol"]
                vars[name] = Symbol(data)
            else
                vars[name] = data
            end
        end
    end
    
    println("Variables loaded from $filename")
    return vars
end

"""
    load_to_workspace(filename::String; overwrite=false)

Load variables from an HDF5 file into the current workspace.
"""
function load_to_workspace(filename::String; overwrite=false)
    vars = read_variables_from_hdf5(filename)
    
    for (name, value) in vars
        var_sym = Symbol(name)
        
        # Skip if variable exists and overwrite=false
        if !overwrite && isdefined(Main, var_sym)
            @warn "Skipping $name: already exists in workspace"
            continue
        end
        
        # Assign to workspace
        Core.eval(Main, :($(var_sym) = $(value)))
    end
    
    println("Loaded $(length(vars)) variables into workspace")
end