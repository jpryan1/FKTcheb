module FactorTime

using Test

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random

r = Sym("r")

rtol          = 1e-3
dct_n         = 100 # Iterations for discrete cosine transform
d             = 3
kern          = 1 / (1+r^2)
lkern         = lambdify(kern)
to            = TimerOutput()
fkt_deg       = 10
mat_kern(x,y) = 1 / (1+norm(x-y)^2)

num_points    = 2000
x_vecs        = [randn(d) / 8 for _ in 1:num_points]
# for idx in 1:length(x_vecs)
#     x_vecs[idx][1] = abs(x_vecs[idx][1])
#     x_vecs[idx][2:end] .= 0
# end
# y_vecs = [randn(d)./8 for _ in 1:num_points]
max_norm  = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))

cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

# Perform FKTcheb
@timeit to "Factor" U_mat, V_mat = degen_kern_harmonic(lkern, x_vecs, x_vecs, cfg)
display(to)
end # module rank_compare
