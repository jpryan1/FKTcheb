module FKTcheb

using LinearAlgebra
using Combinatorics
using TimerOutputs
using LowRankApprox
using StaticArrays
using SpecialFunctions
using SpecialPolynomials
using Polynomials
using Plots

include("factor.jl")
include("util.jl")
include("gegenbauer.jl")
include("hyperspherical.jl")

export get_trans_table, delta, get_pij_table,
    dct, fkt_config, degen_kern_harmonic, gegenbauer,
     A, chebyshev, multiply_polys, integrate_poly,
      evaluate_poly, guess_fkt_err,hyper_normalizer_table,cart2hyp,get_multiindices,hyperspherical

end # module
