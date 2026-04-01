# Run:
# julia --startup-file=no -t 36 --project=scripts scripts/brusselator/brusselator_joint_score.jl --force true

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = get(ENV, "JULIA_PKG_PRECOMPILE_AUTO", "0")
ENV["JULIA_MAKIE_BACKEND"] = get(ENV, "JULIA_MAKIE_BACKEND", "cairomakie")

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "BrusselatorPipeline.jl"))

BrusselatorPipeline.run_cli(:joint_score)
