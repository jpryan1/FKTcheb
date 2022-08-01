module rank_compare

using hdfcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using Plots
using Distributions
using StatsBase

r = Sym("r")
rank_stride = 20
# 3 for 5d gaussian matern
# 20 for 30d 5 for 5d

# high d gaussian n=2k d=40
# σ = 1
# lkern(r)      = exp(-0.5*(r/σ)^2)
# for nystrom_rank in 1:70:(maximum(hdf_ranks)+10)

num_points = 2000
dct_n         = 100 # Iterations for discrete cosine transform
to            = TimerOutput()

d             = 30
#
# kernel_name = "cauchy"
# acc_limit = -15
# σ = 1
# lkern(r)      = 1/(1+(r/σ)^2)
# #
# kernel_name = "gaussian"
# acc_limit = -15
# σ = 1
# lkern(r)      = exp(-0.5*(abs(r)/σ)^2)

# kernel_name = "matern25"
# acc_limit = -15
# σ = 1.5
# lkern(r)      = (1+sqrt(5)*abs(r/σ)+(5/3)*(r/σ)^2)exp(-sqrt(5)*abs(r/σ))

kernel_name="matern15"
acc_limit = -13
σ = 1.5
lkern(r)      = (1+sqrt(3)*abs(r/σ))exp(-sqrt(3)*abs(r/σ))

mat_kern(x,y) = lkern(norm(x-y))
x_vecs    = [randn(d) for _ in 1:num_points]
mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= (mn)
end

#################################################
####     SVD
#################################################

truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
truth_mat_frob_norm = norm(truth_mat)

@timeit to "SVD" svecs, svals = svd(truth_mat);

#################################################
####     HDF
#################################################

hdf_ranks = []
hdf_errs = []

for rtol_pow in -1:-1:acc_limit
    println(rtol_pow)
    rtol = 2.0^rtol_pow
    U_mat, diag_mat = degen_kern_harmonic(lkern, x_vecs, rtol,to)
    V_mat = transpose(U_mat)
    hdf_rank = size(V_mat, 1)
    # if hdf_rank > num_points
    #     break
    # end
    @timeit to "hdf mul" hdf_guess = U_mat*diagm(diag_mat)*V_mat
    if length(hdf_ranks) > 0 && hdf_rank > hdf_ranks[end]+rank_stride
        last_rank = hdf_ranks[end]
        starting_point = hdf_rank
        while starting_point > last_rank+rank_stride
            starting_point -= rank_stride
        end
        for new_rank in starting_point:rank_stride:(hdf_rank-rank_stride)
            push!(hdf_ranks, new_rank)
            new_hdf_guess = U_mat[:,1:new_rank]*diagm(diag_mat[1:new_rank])*V_mat[1:new_rank,:]
            push!(hdf_errs, norm(new_hdf_guess-truth_mat)/truth_mat_frob_norm)
            println(hdf_ranks[end], " ", hdf_errs[end])
        end
    end
    push!(hdf_ranks, hdf_rank)
    push!(hdf_errs, norm(hdf_guess-truth_mat)/truth_mat_frob_norm)
end

# Make errs monotonically decreasing (can revert to more accurate prev form)
min_seen = hdf_errs[1]
for i in 1:length(hdf_errs)
    hdf_errs[i] = min(hdf_errs[i], min_seen)
    global min_seen = min(min_seen,hdf_errs[i])
end

#################################################
####     Set ranks
#################################################

max_svd_rank = maximum(hdf_ranks)
max_rff_rank = maximum(hdf_ranks)

max_nystrom_rank = maximum(hdf_ranks)

#################################################
####     RFF
#################################################

rff_ranks = []
rff_errs = []

rff_u = zeros(num_points, max_rff_rank)
for i in 1:max_rff_rank
    w = randn(d)
    b = rand()*2pi
    for (x_idx, x_vec) in enumerate(x_vecs)
        rff_u[x_idx, i] = sqrt(2.0/max_rff_rank)*cos(dot(w,x_vec)+b)
    end
end
for i in 1:rank_stride:max_rff_rank
    rff_guess = rff_u[:, 1:i]*transpose(rff_u[:,1:i])
    push!(rff_errs, norm(rff_guess-truth_mat)/truth_mat_frob_norm)
    push!(rff_ranks, i)
end

#################################################
####     Nystrom
#################################################

nystrom_ranks = []
nystrom_errs = []
nystrom_lev_errs = []

@timeit to "eig" evecs = eigvecs(truth_mat)
nystrom_perm = randperm(length(x_vecs))
for nystrom_rank in 1:rank_stride:max_nystrom_rank
    if nystrom_rank > num_points
        push!(nystrom_ranks, num_points)
        push!(nystrom_errs, 0)
        push!(nystrom_lev_errs, 0)
        continue
    end
    println("Nystrom rank: ", nystrom_rank)
    q_set = nystrom_perm[1:nystrom_rank]

    @timeit to "popnq " Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
    @timeit to "chol " qmat = cholesky( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
    # Get error for Nystrom
    @timeit to "nysmul " nystrom_guess = Nq * (qmat \ transpose(Nq))
    for i in 1:length(length(x_vecs))
        nystrom_guess[i,i] = truth_mat[i,i]
    end
    push!(nystrom_ranks, nystrom_rank)
    # push!(nystrom_errs, norm(nystrom_guess-truth_mat,2)/svals[1])
    @timeit to "norm " push!(nystrom_errs, norm(nystrom_guess-truth_mat)/truth_mat_frob_norm)

    sub_evecs = copy(evecs[:,1:nystrom_rank])
    lev_scores = [norm(sub_evecs[i, :]) for i in 1:size(sub_evecs,1)]
    q_set = sample(collect(1:num_points), Weights(lev_scores), nystrom_rank, replace=false)
    Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
    qmat = cholesky( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
    nystrom_guess = Nq * (qmat \ transpose(Nq))
    for i in 1:length(length(x_vecs))
        nystrom_guess[i,i] = truth_mat[i,i]
    end
    push!(nystrom_lev_errs, norm(nystrom_guess-truth_mat)/truth_mat_frob_norm)
end

#################################################
####     Output
#################################################

svd_output = open(string("pyplots/data/relerr_vs_rank_svd_", kernel_name, ".txt"), "w")
hdfoutput = open(string("pyplots/data/relerr_vs_rank_hdf_", kernel_name, ".txt"), "w")
nystromoutput = open(string("pyplots/data/relerr_vs_rank_nystrom_", kernel_name, ".txt"), "w")
nystromlevoutput = open(string("pyplots/data/relerr_vs_rank_nystromlev_", kernel_name, ".txt"), "w")
rffoutput = open(string("pyplots/data/relerr_vs_rank_rff_", kernel_name, ".txt"), "w")

for i in 1:max_svd_rank
    if i+1 > num_points
        break
    end
    write(svd_output, string(i,  ",", svals[i+1]/svals[1],"\n"))
end
close(svd_output)

for i in 1:length(rff_ranks)
    write(rffoutput, string(rff_ranks[i],  ",", rff_errs[i],"\n"))
end
close(rffoutput)

for i in 1:length(hdf_ranks)
    write(hdfoutput, string(hdf_ranks[i],  ",", hdf_errs[i],"\n"))
end
close(hdfoutput)

for i in 1:length(nystrom_ranks)
    write(nystromoutput, string(nystrom_ranks[i],  ",", nystrom_errs[i],"\n"))
end
close(nystromoutput)

for i in 1:length(nystrom_ranks)
    write(nystromlevoutput, string(nystrom_ranks[i],  ",", nystrom_lev_errs[i],"\n"))
end
close(nystromlevoutput)

display(to)


# pows = [1, 3, 5, 7, "Adaptive"]
# p = plot(1:100, svals[2:101]/svals[1], yaxis=:log10, label="SVD")
# p = plot!(nystrom_ranks, nystrom_errs, label="Nystrom")
#
# p = plot!(hdf_ranks, hdf_errs, label=string("Pareto"),markershape=:hexagon)
#
# ytick_vals = [10.0^(-i) for i in 1:10]
# xtick_vals = collect(0:50:100)
# p = scatter!([0],[1], label=false, legend=:topright, yticks=(ytick_vals,ytick_vals), xticks=(xtick_vals,xtick_vals))
# display(p)
# savefig(string(spread_param, "_", num_points,"_",dist_name,".pdf"))

end # module rank_compare
