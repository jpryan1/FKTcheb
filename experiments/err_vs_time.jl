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


# Up the trials for matvecs
# Show plots for 50k, 100k, 200k

num_points = 10_000
dct_n         = 100 # Iterations for discrete cosine transform
num_trials=10
d=3
σ = 1
lkern(r)       = 1 / (1+(r/σ)^2)
mat_kern(x,y) = lkern(norm(x-y))
to            = TimerOutput()

# err_results = Dict()

hdfoutput = open("pyplots/data/err_vs_time_hdf_output.txt", "w")

random_samp = randperm(num_points)[1:1000]
x_vecs = [randn(d) for _ in 1:num_points]
mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= mn
end
truth_mat  = mat_kern.(x_vecs[random_samp], permutedims(x_vecs[random_samp]))
truemat_frobnorm = norm(truth_mat,2)
warm_startu, warm_startd =degen_kern_harmonic(lkern, x_vecs, 1e-3, to)

for err_tol_pow in 0:13
    if d > 3 && err_tol_pow < 5 continue end
    GC.gc()
    err_tol = (1e-5)*(2.0^err_tol_pow)
    hdf_t = 0
    hdf_err = 0
    hdf_rank = 0
    mv_time = 0
    for k in 1:num_trials
        GC.gc()
        hdf_t += @elapsed U_mat, diagmat = degen_kern_harmonic(lkern, x_vecs, err_tol, to)
        hdf_rank += size(U_mat, 2)
        hdf_guess = U_mat[random_samp,:]*diagm(diagmat)*transpose(U_mat[random_samp, :])
        hdf_err += (norm(hdf_guess-truth_mat,2)
                        /truemat_frobnorm)
        mv_time += @elapsed res = U_mat*(diagm(diagmat)*(transpose(U_mat)*rand(size(U_mat, 1))))
    end
    hdf_t/=num_trials
    hdf_rank /= num_trials
    hdf_err /= num_trials
    mv_time /=num_trials
    # err_results[(num_points, d, err_tol)] = (hdf_t, hdf_rank, hdf_err, mv_time)
    write(hdfoutput, string(num_points,
     ",", d, ",", err_tol, ",", hdf_t, ",",
      hdf_rank, ",", hdf_err, ",", mv_time, "\n"))
end
close(hdfoutput)

nystromoutput = open("pyplots/data/err_vs_time_nystrom_output.txt", "w")

for idrnk in 10:10:160
    nystrom_t = 0
    nystrom_err = 0
    mv_time = 0
    for k in 1:num_trials
        q_set = randperm(length(x_vecs))[1:idrnk]
        GC.gc()
        nystrom_t += @elapsed begin
            Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
            qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
        end
        nystrom_guess = Nq[random_samp,:] * (qmat \ transpose(Nq[random_samp,:]))
        nystrom_err += norm(nystrom_guess-truth_mat,2)/truemat_frobnorm
        mv_time += @elapsed res = (Nq
            * (qmat \
                (transpose(Nq)*rand(size(Nq, 1)))))
    end
    nystrom_t/=num_trials
    nystrom_err/=num_trials
    mv_time /= num_trials
    write(nystromoutput, string(num_points,
     ",", d, ",", nystrom_t, ",", idrnk, ",",
      nystrom_err, ",", mv_time, "\n"))
end

close(nystromoutput)
