module FKTcheb

using SymPy
using LinearAlgebra
using Combinatorics
using TimerOutputs
using LowRankApprox

include("factor.jl")
include("util.jl")

export fkt_config, cauchy_fkt

end # module
