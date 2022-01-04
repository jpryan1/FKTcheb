module FKTcheb

# using SymPy
using LinearAlgebra
using Combinatorics
using TimerOutputs
using LowRankApprox
using StaticArrays
using SpecialFunctions
using Polynomials
using Plots

include("factor.jl")
include("util.jl")
include("gegenbauer.jl")
include("hyperspherical.jl")

export fkt_config, degen_kern_harmonic, gegenbauer, A, chebyshev, multiply_polys, integrate_poly, evaluate_poly, guess_fkt_err

end # module
