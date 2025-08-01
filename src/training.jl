"""
    generate_txz(y, σ; ϵ=0.05)

Generate training samples with time-dependent noise for reverse diffusion.

# Arguments
- `y`: Original data points
- `σ`: Noise standard deviation function of diffusion time
- `ϵ`: Minimum time value to ensure stability (default: 0.05)

# Returns
- Tuple of (time values, perturbed data, noise vectors)
"""
function generate_txz(y, σ; ϵ=0.05)
    # Generate random time values between ϵ and 1
    t = rand!(similar(y, size(y)[1])) .* (1 - ϵ) .+ ϵ
    
    # Compute noise level at each time point
    σ_t = σ(t)
    
    # Generate random noise and perturb data
    z = randn!(similar(y))
    x = @. y + σ_t * z
    
    return t, x, z
end

"""
    generate_xz(y, σ)

Generate perturbed data and noise vectors for fixed noise level.

# Arguments
- `y`: Original data points
- `σ`: Noise standard deviation (scalar)

# Returns
- Tuple of (perturbed data, noise vectors)
"""
function generate_xz(y, σ)
    z = randn!(similar(y))
    x = @. y + σ * z
    return x, z
end

"""
    generate_data_t(obs, σ; ϵ=0.05)

Generate inputs and targets for time-dependent diffusion training.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `σ`: Noise standard deviation function of diffusion time
- `ϵ`: Minimum time value (default: 0.05)

# Returns
- Tuple of (inputs, targets) where inputs include diffusion time
"""
function generate_data_t(obs, σ; ϵ=0.05)
    # Generate noisy samples and time values
    t, x, z = generate_txz(obs', σ, ϵ=ϵ)
    
    # Pre-allocate arrays for better performance
    n_samples, n_dims = size(obs, 2), size(obs, 1)
    inputs = Matrix{Float32}(undef, n_dims + 1, n_samples)
    targets = Matrix{Float32}(undef, n_dims, n_samples)
    
    # Fill arrays efficiently
    for i in 1:n_samples
        inputs[1:n_dims, i] .= x[i, :]
        inputs[n_dims + 1, i] = t[i]
        targets[:, i] .= z[i, :]
    end
    
    return inputs, targets
end

"""
    generate_data(obs, σ)

Generate inputs and targets for fixed noise level training.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `σ`: Noise standard deviation (scalar)

# Returns
- Tuple of (inputs, targets) for fixed noise level
"""
function generate_data(obs, σ)
    # Generate noisy samples
    x, z = generate_xz(obs', σ)
    
    # Pre-allocate arrays for better performance
    n_samples, n_dims = size(obs, 2), size(obs, 1)
    inputs = Matrix{Float32}(undef, n_dims, n_samples)
    targets = Matrix{Float32}(undef, n_dims, n_samples)
    
    # Fill arrays efficiently
    for i in 1:n_samples
        inputs[:, i] .= x[i, :]
        targets[:, i] .= z[i, :]
    end
    
    return inputs, targets
end

"""
    swish(x)

Swish activation function: x * sigmoid(x)

# Arguments
- `x`: Input value

# Returns
- Activated value
"""
function swish(x)
    return x * sigmoid(x)
end

"""
    create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)

Create a neural network with the given architecture.

# Arguments
- `neurons`: Vector of layer sizes (including input and output dimensions)
- `activation`: Activation function for hidden layers (default: swish)
- `last_activation`: Activation function for output layer (default: identity)

# Returns
- Flux Chain neural network model
"""
function create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)
    layers = Vector{Any}(undef, length(neurons) - 1)
    
    # Create hidden layers with specified activation
    for i in 1:length(neurons)-2
        layers[i] = Flux.Dense(neurons[i], neurons[i+1], activation)
    end
    
    # Create output layer with specified activation
    layers[end] = Flux.Dense(neurons[end-1], neurons[end], last_activation)
    
    return Flux.Chain(layers...)
end

"""
    loss_score(nn, inputs, targets)

Compute mean squared error loss between model predictions and targets.

# Arguments
- `nn`: Neural network model
- `inputs`: Input features
- `targets`: Target values

# Returns
- Mean squared error loss
"""
function loss_score(nn, inputs, targets)
    predictions = nn(inputs)  
    return Flux.mse(predictions, targets) 
end

"""
    train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ; kwargs...)

Train a new neural network with vanilla score matching loss.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `n_epochs`: Number of training epochs
- `batch_size`: Mini-batch size for training
- `neurons`: Vector of layer sizes
- `σ`: Noise standard deviation (scalar or function)
- `opt`: Optimizer (default: Adam(0.001))
- `activation`: Activation function (default: swish)
- `last_activation`: Output activation (default: identity)
- `ϵ`: Minimum time value (default: 0.05)
- `use_gpu`: Whether to use GPU acceleration (default: true)

# Returns
- Tuple of (trained model, loss history)
"""
function train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ; 
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity, 
               ϵ=0.05, use_gpu=true)
    
    # Setup compute device
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    
    # Create model and move to device
    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device
    
    # Set up optimizer state using modern Flux API
    opt_state = Flux.setup(opt, nn)
    
    # Initialize loss history
    losses = []
    
    # Training loop
    for epoch in ProgressBar(1:n_epochs) 
        # Generate batch data based on sampling method
        if isa(σ, Float64)      # Langevin sampling method
            inputs, targets = generate_data(obs, σ)
        else                    # Reverse sampling method
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        
        # Create data loader
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        
        # Mini-batch updates
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            
            # Compute gradients and loss using modern API
            loss, grads = Flux.withgradient(nn) do m
                loss_score(m, batch_inputs, batch_targets)
            end
            
            # Update parameters using modern API
            Flux.update!(opt_state, nn, grads[1])
            
            # Accumulate loss
            epoch_loss += loss
        end
        
        # Record average loss for this epoch
        push!(losses, epoch_loss / length(data_loader))
    end
    
    return nn, losses
end

function train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ, time_dim; 
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity, 
               ϵ=0.05, use_gpu=true)
    
    # Setup compute device
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    
    # Create model and move to device
    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device
    
    # Set up optimizer state using modern Flux API
    opt_state = Flux.setup(opt, nn)
    
    # Initialize loss history
    losses = []
    
    # Training loop
    for epoch in ProgressBar(1:n_epochs) 
        # Generate batch data based on sampling method
        if isa(σ, Float64)      # Langevin sampling method
            inputs, targets = generate_data(obs, σ)
        else                    # Reverse sampling method
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        
        # Create data loader
        data_loader = Flux.DataLoader((inputs, targets[time_dim+1:end, :]), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        
        # Mini-batch updates
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            
            # Compute gradients and loss using modern API
            loss, grads = Flux.withgradient(nn) do m
                loss_score(m, batch_inputs, batch_targets)
            end
            
            # Update parameters using modern API
            Flux.update!(opt_state, nn, grads[1])
            
            # Accumulate loss
            epoch_loss += loss
        end
        
        # Record average loss for this epoch
        push!(losses, epoch_loss / length(data_loader))
    end
    
    return nn, losses
end

"""
    train(obs, n_epochs, batch_size, nn::Chain, σ; kwargs...)

Continue training an existing neural network with vanilla score matching loss.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `n_epochs`: Number of training epochs
- `batch_size`: Mini-batch size for training
- `nn`: Existing neural network model
- `σ`: Noise standard deviation (scalar or function)
- `opt`: Optimizer (default: Adam(0.001))
- `ϵ`: Minimum time value (default: 0.05)
- `use_gpu`: Whether to use GPU acceleration (default: true)

# Returns
- Tuple of (trained model, loss history)
"""
function train(obs, n_epochs, batch_size, nn::Chain, σ; 
               opt=Flux.Adam(0.001), ϵ=0.05, use_gpu=true)
    
    # Setup compute device
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    
    # Move model to device
    nn = nn |> device
    
    # Set up optimizer state using modern Flux API
    opt_state = Flux.setup(opt, nn)
    
    # Initialize loss history
    losses = []
    
    # Training loop
    for epoch in ProgressBar(1:n_epochs) 
        # Generate batch data based on sampling method
        if isa(σ, Float64)     # Langevin sampling method
            inputs, targets = generate_data(obs, σ)
        else                   # Reverse sampling method
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        
        # Create data loader
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        
        # Mini-batch updates
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            
            # Compute gradients and loss using modern API
            loss, grads = Flux.withgradient(nn) do m
                loss_score(m, batch_inputs, batch_targets)
            end
            
            # Update parameters using modern API
            Flux.update!(opt_state, nn, grads[1])
            
            # Accumulate loss
            epoch_loss += loss
        end
        
        # Record average loss for this epoch
        push!(losses, epoch_loss / length(data_loader))
    end
    
    return nn, losses
end

"""
    train(obs, n_epochs, batch_size, neurons; kwargs...)

Train a neural network with pre-clustered data.

# Arguments
- `obs`: Tuple of (inputs, targets) for pre-clustered data
- `n_epochs`: Number of training epochs
- `batch_size`: Mini-batch size for training
- `neurons`: Vector of layer sizes
- `opt`: Optimizer (default: Adam(0.001))
- `activation`: Activation function (default: swish)
- `last_activation`: Output activation (default: identity)
- `use_gpu`: Whether to use GPU acceleration (default: true)

# Returns
- Tuple of (trained model, loss history)
"""
function train(obs, n_epochs, batch_size, neurons; 
               opt=Flux.Adam(0.001), activation=swish, last_activation=identity, use_gpu=true)
    
    # Setup compute device
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    
    # Create model and move to device
    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device
    
    # Set up optimizer state using Flux's new API
    opt_state = Flux.setup(opt, nn)
    
    # Initialize loss history
    losses = []
    
    # Convert to Float32 once before training loop
    inputs, targets = obs
    inputs = Float32.(inputs)
    targets = Float32.(targets)
    
    # Training loop
    for epoch in ProgressBar(1:n_epochs) 
        # Create data loader (data already converted to Float32)
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        
        # Mini-batch updates
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            
            # Compute gradients and loss using the new API
            loss, grads = Flux.withgradient(nn) do m
                loss_score(m, batch_inputs, batch_targets)
            end
            
            # Update parameters using the new API
            Flux.update!(opt_state, nn, grads[1])
            
            # Accumulate loss
            epoch_loss += loss
        end
        
        # Record average loss for this epoch
        push!(losses, epoch_loss / length(data_loader))
    end
    
    return nn, losses
end

"""
    check_loss(obs, nn, σ; ϵ=0.05, n_samples=1)

Evaluate loss on random batches of data.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `nn`: Neural network model
- `σ`: Noise standard deviation (scalar or function)
- `ϵ`: Minimum time value (default: 0.05)
- `n_samples`: Number of evaluation batches (default: 1)

# Returns
- Average loss across evaluation batches
"""
function check_loss(obs, nn, σ; ϵ=0.05, n_samples=1)
    loss = 0.0
    
    # Average loss over multiple samples
    for _ in ProgressBar(1:n_samples) 
        # Generate evaluation data
        if isa(σ, Float64)
            inputs, targets = generate_data(obs, σ)
        else
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        
        # Accumulate loss
        loss += loss_score(nn, inputs, targets)
    end
    
    return loss/n_samples
end

"""
    train_with_early_stopping(obs, n_epochs, batch_size, neurons::Vector{Int}, σ; 
                             validation_split=0.1, patience=10, kwargs...)

Train with early stopping based on validation loss.

# Arguments
- `obs`: Original observations with shape (dim, n_samples)
- `n_epochs`: Maximum number of training epochs
- `batch_size`: Mini-batch size for training
- `neurons`: Vector of layer sizes
- `σ`: Noise standard deviation (scalar or function)
- `validation_split`: Fraction of data to use for validation (default: 0.1)
- `patience`: Number of epochs with no improvement before stopping (default: 10)
- Additional keyword arguments passed to the main train function

# Returns
- Tuple of (best model, training loss history, validation loss history)
"""
function train_with_early_stopping(obs, n_epochs, batch_size, neurons::Vector{Int}, σ; 
                                  validation_split=0.1, patience=10, kwargs...)
    
    # Split data into training and validation sets
    n_samples = size(obs, 2)
    n_val = Int(floor(n_samples * validation_split))
    indices = randperm(n_samples)
    
    val_indices = indices[1:n_val]
    train_indices = indices[n_val+1:end]
    
    train_obs = obs[:, train_indices]
    val_obs = obs[:, val_indices]
    
    # Initialize model
    nn = create_nn(neurons; kwargs...)
    
    # Initialize tracking variables
    best_val_loss = Inf
    best_model = deepcopy(nn)
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    # Training loop with early stopping
    for epoch in 1:n_epochs
        # Train for one epoch
        nn, epoch_loss = train(train_obs, 1, batch_size, nn, σ; kwargs...)
        push!(train_losses, epoch_loss[1])
        
        # Evaluate on validation set
        val_loss = check_loss(val_obs, nn, σ; n_samples=1)
        push!(val_losses, val_loss)
        
        println("Epoch $epoch: Train loss = $(epoch_loss[1]), Val loss = $val_loss")
        
        # Check if validation loss improved
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_model = deepcopy(nn)
            epochs_no_improve = 0
        else
            epochs_no_improve += 1
            if epochs_no_improve >= patience
                println("Early stopping triggered after $epoch epochs")
                break
            end
        end
    end
    
    return best_model, train_losses, val_losses
end