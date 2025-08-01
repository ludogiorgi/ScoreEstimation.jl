"""
    sample_reverse(Dim, nn, n_samples, n_diffs, σ, g)

Sample points using reverse diffusion process.

# Arguments
- `Dim`: Dimension of state space
- `nn`: Neural network model for score estimation
- `n_samples`: Number of samples to generate
- `n_diffs`: Number of diffusion steps to reverse
- `σ`: Noise standard deviation function of diffusion time
- `g`: Drift coefficient function of diffusion time

# Returns
- Matrix of generated samples with shape (Dim, n_samples)
"""
function sample_reverse(Dim, nn, n_samples, n_diffs, σ, g)
    # Time step for discretization
    dt = 1.0 / n_diffs
    
    # Initialize ensemble of samples
    ens = zeros(Dim, n_samples)
    
    # Generate each sample independently
    for i in 1:n_samples
        # Start from noise distribution at t=1
        xOld = σ(1) * randn(Dim)
        
        # Reverse diffusion process
        for t in 1:n_diffs
            # Current diffusion time (decreasing from 1 to 0)
            t_diff = (n_diffs - t + 1) / n_diffs
            
            # Get noise level at current time
            s = σ(t_diff)
            
            # Estimate score function and normalize
            score = nn([xOld..., t_diff]) ./ s
            
            # Update step using reverse SDE
            xNew = xOld .+ score .* g(t_diff)^2 .* dt .+ randn(Dim) .* sqrt(dt) .* g(t_diff)
            xOld = xNew
        end
        
        # Store final sample
        ens[:,i] = xOld
    end
    
    return ens
end

"""
    sample_langevin(T, dt, f, obs; seed=123, res=1, boundary=false)

Sample points using Langevin dynamics with a given score function.

# Arguments
- `T`: Total simulation time
- `dt`: Time step size
- `f`: Score function (gradient of log probability)
- `obs`: Observation data for initialization and boundary reset
- `seed`: Random seed for reproducibility (default: 123)
- `res`: Save results every `res` steps (default: 1)
- `boundary`: If specified as [min, max], resets to random observation when exceeded (default: false)

# Returns
- Matrix of generated trajectory with shape (dim, num_saved_steps)
"""
function sample_langevin(T, dt, f, obs; seed=123, res=1, boundary=false)
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Calculate number of steps and dimensions
    N = Int(T / dt)
    dim = length(obs[:,1])
    num_saved_steps = ceil(Int, N / res)
    
    # Initialize storage and trajectory
    x = zeros(dim, num_saved_steps)
    idx = 1
    
    # Start from random observation
    x0 = obs[:,rand(1:length(obs[1,:]))]
    x_temp = x0
    
    # Counter for boundary crossings
    count = 0
    
    # Langevin dynamics integration
    for t in ProgressBar(2:N)
        # Deterministic step (gradient of log probability)
        rk4_step!(x_temp, dt, f)
        
        # Stochastic step
        x_temp += √2 * randn(dim) * sqrt(dt)
        
        # Handle boundary conditions if specified
        if boundary != false
            if any(x_temp .< boundary[1]) || any(x_temp .> boundary[2])
                x_temp = obs[:,rand(1:length(obs[1,:]))]
                count += 1
            end
        end
        
        # Save results at specified resolution
        if t % res == 0
            x[:, idx] = x_temp
            idx += 1
        end
    end
    
    println("Number of boundary crossings: ", count)
    return x
end

"""
    sample_langevin_Σ(Nsteps, dt, f, obs, Σ; seed=123, res=1, boundary=false)

Sample points using Langevin dynamics with custom diffusion matrix.

# Arguments
- `Nsteps`: Number of simulation steps
- `dt`: Time step size
- `f`: Score function (gradient of log probability)
- `obs`: Observation data for initialization and boundary reset
- `Σ`: Diffusion matrix
- `seed`: Random seed for reproducibility (default: 123)
- `res`: Save results every `res` steps (default: 1)
- `boundary`: If specified as [min, max], resets to random observation when exceeded (default: false)

# Returns
- Matrix of generated trajectory with shape (dim, num_saved_steps)
"""
function sample_langevin_Σ(Nsteps, dt, f, obs, Σ; seed=123, res=1, boundary=false)
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Calculate dimensions
    dim = length(obs[:,1])
    num_saved_steps = ceil(Int, Nsteps / res)
    
    # Initialize storage and trajectory
    x = zeros(dim, num_saved_steps)
    idx = 1
    
    # Start from random observation
    x0 = obs[:,rand(1:length(obs[1,:]))]
    x_temp = x0
    
    # Counter for boundary crossings
    count = 0
    
    # Langevin dynamics integration
    for t in ProgressBar(2:Nsteps)
        # Create modified drift function that incorporates diffusion matrix
        f_Σ2(x) = (Σ * Σ') * f(x)
        
        # Deterministic step
        rk4_step!(x_temp, dt, f_Σ2)
        
        # Stochastic step with correlated noise
        x_temp += Σ * randn(dim) * sqrt(2dt)
        
        # Handle boundary conditions if specified
        if boundary != false
            if any(x_temp .< boundary[1]) || any(x_temp .> boundary[2])
                x_temp = obs[:,rand(1:length(obs[1,:]))]
                count += 1
            end
        end
        
        # Save results at specified resolution
        if t % res == 0
            x[:, idx] = x_temp
            idx += 1
        end
    end
    
    println("Probability of boundary crossings: ", count/Nsteps)
    return x
end

"""
    sample_reverse_conditional(Dim, nn, n_samples, n_diffs, σ, g, condition_idx, condition_val)

Sample points using reverse diffusion with conditional constraints.

# Arguments
- `Dim`: Dimension of state space
- `nn`: Neural network model for score estimation
- `n_samples`: Number of samples to generate
- `n_diffs`: Number of diffusion steps to reverse
- `σ`: Noise standard deviation function of diffusion time
- `g`: Drift coefficient function of diffusion time
- `condition_idx`: Indices of dimensions to condition on
- `condition_val`: Target values for conditioned dimensions

# Returns
- Matrix of conditionally generated samples with shape (Dim, n_samples)
"""
function sample_reverse_conditional(Dim, nn, n_samples, n_diffs, σ, g, condition_idx, condition_val)
    # Time step for discretization
    dt = 1.0 / n_diffs
    
    # Initialize ensemble of samples
    ens = zeros(Dim, n_samples)
    
    # Generate each sample independently
    for i in 1:n_samples
        # Start from noise distribution at t=1
        xOld = σ(1) * randn(Dim)
        
        # Apply initial conditioning by setting specified dimensions to target values
        xOld[condition_idx] = condition_val
        
        # Reverse diffusion process
        for t in 1:n_diffs
            # Current diffusion time (decreasing from 1 to 0)
            t_diff = (n_diffs - t + 1) / n_diffs
            
            # Get noise level at current time
            s = σ(t_diff)
            
            # Estimate score function and normalize
            score = nn([xOld..., t_diff]) ./ s
            
            # Update step using reverse SDE
            xNew = xOld .+ score .* g(t_diff)^2 .* dt .+ randn(Dim) .* sqrt(dt) .* g(t_diff)
            
            # Maintain conditioning by resetting specified dimensions
            xNew[condition_idx] = condition_val
            
            xOld = xNew
        end
        
        # Store final sample
        ens[:,i] = xOld
    end
    
    return ens
end

"""
    ensemble_sampler(Dim, nn, n_samples, n_diffs, σ, g; batch_size=100)

Generate samples in batches for improved efficiency.

# Arguments
- `Dim`: Dimension of state space
- `nn`: Neural network model for score estimation
- `n_samples`: Total number of samples to generate
- `n_diffs`: Number of diffusion steps to reverse
- `σ`: Noise standard deviation function of diffusion time
- `g`: Drift coefficient function of diffusion time
- `batch_size`: Number of samples to generate in each batch (default: 100)

# Returns
- Matrix of generated samples with shape (Dim, n_samples)
"""
function ensemble_sampler(Dim, nn, n_samples, n_diffs, σ, g; batch_size=100)
    # Initialize results array
    ens = zeros(Dim, n_samples)
    
    # Calculate number of full batches and remaining samples
    n_batches = div(n_samples, batch_size)
    remainder = mod(n_samples, batch_size)
    
    # Generate samples in batches
    for i in 1:n_batches
        idx_range = ((i-1)*batch_size+1):(i*batch_size)
        ens[:, idx_range] = sample_reverse(Dim, nn, batch_size, n_diffs, σ, g)
        println("Generated batch $i of $n_batches")
    end
    
    # Generate remaining samples
    if remainder > 0
        idx_range = (n_batches*batch_size+1):n_samples
        ens[:, idx_range] = sample_reverse(Dim, nn, remainder, n_diffs, σ, g)
        println("Generated final batch of $remainder samples")
    end
    
    return ens
end