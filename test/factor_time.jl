module FactorTime

using Test

using hdfcheb
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random

GC.gc()

d             = 7
rtol          = 1e-3
num_points    = 10_000

random_samp = randperm(num_points)[1:1000]


dct_n         = 100 # Iterations for discrete cosine transform
to            = TimerOutput()
σ = 2
lkern(r)      = 1/(1+(σ*r)^2)
mat_kern(x,y) = lkern(norm(x-y))

x_vecs    = [randn(d) for _ in 1:num_points]
mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= (mn)
end
truth_mat  = mat_kern.(x_vecs[random_samp], permutedims(x_vecs[random_samp]))
truemat_frobnorm = norm(truth_mat,2)

# truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
# Perform hdfcheb
GC.gc()
@timeit to "Factor" U_mat,diag_mat = degen_kern_harmonic(lkern, x_vecs, rtol,to)
# V_mat = transpose(U_mat)
hdf_rank = size(U_mat, 2)
hdf_guess = (U_mat*diagm(diag_mat)*transpose(U_mat))
# hdf_err_2norm = norm(hdf_guess-truth_mat, 2)/norm(truth_mat,2)
# println("Rank ", hdf_rank)
# println("Rtol ", rtol)
# println("Num points ", num_points)
# println("Err ",hdf_err_2norm)

GC.gc()
q_set = randperm(length(x_vecs))[1:hdf_rank]
# possibly priority to high norm columns
# Leverage score


@timeit to string("Nystrom ") begin
    Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
    qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
end
x_vecs    = [randn(d) for _ in 1:num_points]
mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= (mn)
end
q_set = randperm(length(x_vecs))[1:hdf_rank]
@timeit to "Factor2" U_mat,diag_mat = degen_kern_harmonic(lkern, x_vecs, rtol,to)

@timeit to string("Nystrom2 ") begin

timer = @elapsed begin
    Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
    qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
end
end
println(timer, " Time")
GC.gc()

x = randn(num_points)
@timeit to "nystrom matvec" bp = Nq * (qmat \ (transpose(Nq) * x))
GC.gc()

@timeit to "hdf matvec" b = U_mat * (transpose(U_mat)*x)

nystrom_guess = Nq[random_samp,:] * (qmat \ transpose(Nq[random_samp,:]))
nystrom_err = norm(nystrom_guess-truth_mat,2)/truemat_frobnorm

println(nystrom_err)
println("RANK :, ", hdf_rank)
display(to)
end # module rank_compare
