#=
   NOISING SCHEDULES MODULE
   
   This module defines various noise scheduling functions used in diffusion and 
   score-based generative models. These schedules control how noise is added or 
   removed during the forward and reverse processes.
=#

"""
    σ_variance_exploding(t; σ_min = 0.01, σ_max = 1.0)

Variance exploding noise schedule that increases exponentially with time.

# Arguments
- `t`: Time parameter between 0 and 1
- `σ_min`: Minimum noise level at t = 0
- `σ_max`: Maximum noise level at t = 1

# Returns
- Noise standard deviation at time t
"""
σ_variance_exploding(t; σ_min = 0.01, σ_max = 1.0) = @. σ_min * (σ_max/σ_min)^t

"""
    g_variance_exploding(t; σ_min=0.01, σ_max=1.0)

Computes the drift coefficient for the variance exploding SDE.
This is the derivative of σ(t) multiplied by scaling factors needed in the diffusion process.

# Arguments
- `t`: Time parameter between 0 and 1
- `σ_min`: Minimum noise level at t = 0
- `σ_max`: Maximum noise level at t = 1

# Returns
- Drift coefficient at time t
"""
g_variance_exploding(t; σ_min=0.01, σ_max=1.0) = σ_min * (σ_max/σ_min)^t * sqrt(2*log(σ_max/σ_min))

"""
    σ_variance_preserving(t; β_min = 0.1, β_max = 20.0)

Variance preserving noise schedule based on the integration of a noise schedule β(t).

# Arguments
- `t`: Time parameter between 0 and 1
- `β_min`: Minimum noise intensity at t = 0
- `β_max`: Maximum noise intensity at t = 1

# Returns
- Noise standard deviation at time t
"""
function σ_variance_preserving(t; β_min = 0.1, β_max = 20.0)
    # Linear β schedule
    β_t = β_min + t * (β_max - β_min)
    
    # Compute variance from β integration
    α_t = exp(-0.5 * β_t)
    
    # Return standard deviation
    return sqrt(1 - α_t^2)
end

"""
    g_variance_preserving(t; β_min = 0.1, β_max = 20.0)

Computes the drift coefficient for the variance preserving SDE.

# Arguments
- `t`: Time parameter between 0 and 1
- `β_min`: Minimum noise intensity at t = 0
- `β_max`: Maximum noise intensity at t = 1

# Returns
- Drift coefficient at time t
"""
function g_variance_preserving(t; β_min = 0.1, β_max = 20.0)
    # Linear β schedule
    β_t = β_min + t * (β_max - β_min)
    
    return sqrt(β_t)
end

# Export the main functions
export σ_variance_exploding, g_variance_exploding
export σ_variance_preserving, g_variance_preserving