module FactorTime

using Test

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random

GC.gc()
r = Sym("r")

d             = 3
num_points    = 100_000
rtol = (1e-4)


spread_param  = 1 # 2 for unif, 6 for norm
dct_n         = 100 # Iterations for discrete cosine transform
σ = 2
lkern(r)      = 1 / (1+(σ*r)^2)
mat_kern(x,y) = lkern(norm(x-y))
to            = TimerOutput()

x_vecs        = [randn(d) for _ in 1:num_points]
centroid = sum(x_vecs)/length(x_vecs)
for i in 1:length(x_vecs)
    x_vecs[i] .-= centroid
end

mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= (mn)
    x_vecs[i] *= (spread_param)
    # x_vecs[i] /= norm(x_vecs[i])
end

@timeit to "Factor" U_mat, diag = degen_kern_harmonic(lkern, x_vecs, rtol,to)
fkt_rank = size(U_mat, 2)
println("Rank ", fkt_rank)
println("Rtol ", rtol)
println("Num points ", num_points)
# println("Trunc param ", best_deg)

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
