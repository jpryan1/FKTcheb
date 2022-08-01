using hdfcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using Distributions
using StaticArrays
using Polynomials
using SpecialFunctions
using Combinatorics
using LaTeXStrings

dct_n         = 100 # Iterations for discrete cosine transform
err_tol       = (3e-3)
σ = 1
lkern(r)       = 1 / (1+(r/σ)^2)
mat_kern(x,y) = 1 / (1+norm(x-y)^2)
to            = TimerOutput()

hdfoutput = open("pyplots/data/time_vs_n_output.txt", "w")


for num_points_pow in 16:20
    num_points = 2^(num_points_pow)
    for d in 3:4:15
        x_vecs = [randn(d)  for _ in 1:num_points]
        mn = maximum(norm.(x_vecs))
        for i in 1:length(x_vecs)
            x_vecs[i] /= mn
        end
        hdf_t = 0
        for k in 1:10
            GC.gc()
            hdf_t += @elapsed U_mat, diagmat = degen_kern_harmonic(lkern, x_vecs, err_tol, to)
            # println(size(U_mat,2))
        end
        if num_points_pow == 16 && d == 3  # warmup
            hdf_t = 0
            for k in 1:10
                GC.gc()
                hdf_t += @elapsed U_mat, diagmat = degen_kern_harmonic(lkern, x_vecs, err_tol, to)
            end
        end
        write(hdfoutput, string(num_points,  ",", d,",", hdf_t/10, "\n"))
    end
end
close(hdfoutput)
