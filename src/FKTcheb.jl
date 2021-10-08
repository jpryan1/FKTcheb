module FKTcheb

using SymPy
using LinearAlgebra
using Combinatorics
using TimerOutputs
using LowRankApprox
using StaticArrays
using SpecialFunctions

include("factor.jl")
include("util.jl")
include("gegenbauer.jl")
include("hyperspherical.jl")

export fkt_config, cheb_fkt

end # module
