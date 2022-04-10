module hdfcheb

using SymPy
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

export hdf_config,hyperspherical, hypospherical,gegenbauer_normalizer, degen_kern_harmonic, get_multiindices,gegenbauer,hyper_normalizer_table, A, chebyshev, multiply_polys, integrate_poly, evaluate_poly, guess_hdf_err, cart2hyp

end # module
