#!/usr/bin/env julia

include(joinpath(@__DIR__, "CoupledL63Pipeline.jl"))

using .CoupledL63Pipeline

CoupledL63Pipeline.main()
