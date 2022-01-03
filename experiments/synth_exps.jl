module rank_compare

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using Plots
using Distributions

r = Sym("r")


# Control params

spread_param = 6
dct_n         = 100 # Iterations for discrete cosine transform
kern          = 1 / (1+r^2)
mat_kern(x,y) = 1 / (1+norm(x-y)^2)
lkern         = lambdify(kern)
to            = TimerOutput()

dist_name = "normal"
num_points = 1000
d          = 3

results = Dict()
for num_points in 50000:50000:250000
    random_samp = randperm(num_points)[1:1000]
    for d in 3:4:15
        x_vecs = [randn(d) / spread_param for _ in 1:num_points]
        truth_mat  = mat_kern.(x_vecs[random_samp], permutedims(x_vecs[random_samp]))
        truemat_frobnorm = norm(truth_mat)
        for fkt_deg in 2:2:10
            GC.gc()
            println((num_points, d, fkt_deg))
            rtol = 10.0^(-15)
            cfg = fkt_config(fkt_deg, d, dct_n, to, rtol)
            rtol = guess_fkt_err(lkern, x_vecs, cfg)
            cfg = fkt_config(fkt_deg, d, dct_n, to, rtol)
            fkt_t = @elapsed U_mat = degen_kern_harmonic(lkern, x_vecs, cfg)
            V_mat = transpose(U_mat)
            fkt_rank = size(U_mat, 2)

            fkt_guess = U_mat[random_samp,:]*V_mat[:, random_samp]
            fkt_err = (norm(fkt_guess-truth_mat)
                            /truemat_frobnorm)
            results[(num_points, d, fkt_deg)] = (fkt_t, fkt_rank, fkt_err)
        end
    end
end

println(results)

end # module rank_compare
