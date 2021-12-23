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

rtol          = 1e-2
dct_n         = 100 # Iterations for discrete cosine transform
d             = 4
kern          = 1 / (1+r^2)
lkern         = lambdify(kern)
to            = TimerOutput()
fkt_deg       = 8
mat_kern(x,y) = 1 / (1+norm(x-y)^2)

num_points    = 400000
x_vecs        = [randn(d) / 6 for _ in 1:num_points]
# for idx in 1:length(x_vecs)
#     x_vecs[idx][1] = abs(x_vecs[idx][1])
#     x_vecs[idx][2:end] .= 0
# end
# y_vecs = [randn(d)./8 for _ in 1:num_points]
max_norm  = maximum(norm.(x_vecs))
# truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
# _, svals = svd(truth_mat);

cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)


# Perform FKTcheb
@timeit to "Factor" U_mat = degen_kern_harmonic(lkern, x_vecs, cfg)
# V_mat = transpose(U_mat)
fkt_rank = size(U_mat, 2)
# fkt_guess = (U_mat*V_mat)
# fkt_err_2norm = norm(fkt_guess-truth_mat, 2)/svals[1]
println("Rank ", fkt_rank)
# println("Err ",fkt_err_2norm)

GC.gc()
q_set = randperm(length(x_vecs))[1:fkt_rank]
# possibly priority to high norm columns
# Leverage score
@timeit to string("Nystrom ") begin
    Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
    qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
end
GC.gc()

x = randn(num_points)
@timeit to "nystrom matvec" bp = Nq * (qmat \ (transpose(Nq) * x))
GC.gc()

@timeit to "fkt matvec" b = U_mat * (transpose(U_mat)*x)

# nystrom_guess = Nq * (qmat \ transpose(Nq))
# nystrom_err_2norm = norm(nystrom_guess-truth_mat,2)/svals[1]
# println("Nystrom err ", nystrom_err_2norm)


display(to)
end # module rank_compare
