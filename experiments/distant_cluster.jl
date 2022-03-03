module rank_compare

using FKTcheb
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using Plots
using Distributions


spread_param = 6
num_points = 1000
dct_n         = 100 # Iterations for discrete cosine transform
d             = 3
lkern(r)          = exp(-r^2)
mat_kern(x,y) = lkern(norm(x-y))
to            = TimerOutput()

x_vecs = [randn(d) / spread_param for _ in 1:num_points]
truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))

svecs, svals = svd(truth_mat);

fkt_ranks = []
fkt_errs = []

# for tolpow in 1:2:7
#
#     new_fkt_ranks = []
#     new_fkt_errs = []
#     for fkt_deg in 2:2:10
#         println("FKT test: ",tolpow, " ", fkt_deg)
#         rtol = 10.0^(-tolpow)
#
#         cfg = fkt_config(fkt_deg, d, dct_n, to, rtol)
#         U_mat = degen_kern_harmonic(lkern, x_vecs, cfg)
#         # println("U_mat size ", size(U_mat))
#         V_mat = transpose(U_mat)
#         fkt_rank = size(V_mat, 1)
#         if fkt_rank > 100
#             break
#         end
#         fkt_guess = U_mat*V_mat
#         # _, fkterrorsvals = svd(fkt_guess-truth_mat)
#         # println("Rel err ",fkt_err_2norm/svals[1])
#         push!(new_fkt_ranks, fkt_rank)
#         # push!(new_fkt_errs, fkterrorsvals[1]/svals[1])
#         push!(new_fkt_errs, norm(fkt_guess-truth_mat, 2)/svals[1])
#     end
#     push!(fkt_ranks, new_fkt_ranks)
#     push!(fkt_errs, new_fkt_errs)
# end


# PARETO CURVE
new_fkt_ranks = []
new_fkt_errs = []
for err_tol_pow in 0:9
    err_tol = (1e-5)*(2.0^err_tol_pow)
    U_mat,diag = degen_kern_harmonic(lkern, x_vecs, err_tol,to)
    V_mat = transpose(U_mat)
    fkt_rank = size(V_mat, 1)
    # if fkt_rank > 100
    #     break
    # end
    fkt_guess = U_mat*diagm(diag)*V_mat
    # _, fkterrorsvals = svd(fkt_guess-truth_mat)
    fkt_err_2norm = norm(fkt_guess-truth_mat, 2)
    push!(new_fkt_ranks, fkt_rank)
    # push!(new_fkt_errs, fkterrorsvals[1]/svals[1])
    push!(new_fkt_errs, fkt_err_2norm/svals[1])
    # push!(new_fkt_errs, rtol)
end
push!(fkt_ranks, new_fkt_ranks)
push!(fkt_errs, new_fkt_errs)


nystrom_ranks = []
nystrom_errs = []

for idrnk in 1:10:100
    println("Nystrom rank: ",idrnk)
    q_set = randperm(length(x_vecs))[1:idrnk]
    # possibly priority to high norm columns
    # Leverage score
    @timeit to string("Nystrom ", length(x_vecs)) begin
        Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
        qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
    end

    # Get error for Nystrom
    nystrom_guess = Nq * (qmat \ transpose(Nq))
    for i in 1:length(length(x_vecs))
        nystrom_guess[i,i] = truth_mat[i,i]
    end
    push!(nystrom_ranks, idrnk)
    push!(nystrom_errs, norm(nystrom_guess-truth_mat,2)/svals[1])
    # println("Nystrom r=",idrnk, ", err=", nystromerror)
end

pows = ["Adaptive"]
p = plot(1:100, svals[2:101]/svals[1], yaxis=:log10, label="SVD")
p = plot!(nystrom_ranks, nystrom_errs, label="Nystrom")
for i in 1:(length(fkt_ranks)-1)
    p = plot!(fkt_ranks[i], fkt_errs[i], label=string("FKT ", pows[i]))
end
p = plot!(fkt_ranks[end], fkt_errs[end], label=string("FKT"),markershape=:hexagon, thickness_scaling = 1.25,xlabel="Rank", ylabel="Rel err")

ytick_vals = [10.0^(-i) for i in 1:10]
xtick_vals = collect(0:50:100)
p = scatter!([0],[1], label=false, legend=:topright, yticks=(ytick_vals,ytick_vals), xticks=(xtick_vals,xtick_vals))
display(p)
# savefig(string(spread_param, "_", num_points,"_",dist_name,".pdf"))



#
#     fktid = idfact(guess, rtol = error)
#     println("looks like rank ", length(fktid.sk))
#     # Perform IDs for FKTcheb+
#     @timeit to string("FKT+ IDs ", num_points) begin
#         U_mat_id = idfact(U_mat, rtol = error)
#         V_mat_id = idfact(V_mat, rtol = error)
#     end
#
#     # Get error for FKTcheb+
#     U_mat[:, U_mat_id.rd] .= U_mat[:, U_mat_id.sk]*U_mat_id.T
#     V_mat[:, V_mat_id.rd] .= V_mat[:, V_mat_id.sk]*V_mat_id.T
#     guess = U_mat*V_mat
#     errorplus = norm(guess-truth_mat)/norm(truth_mat)
#     println("FKT+ Error: ", errorplus)
#
#
#     # Perform ID with tolerance for same error as FKTcheb
#     @timeit to string("ID ",num_points) begin
#         id_full_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
#         idf = idfact(id_full_mat, rtol = error)
#     end
#
#     # Get error for ID
#     id_full_mat[:, idf.rd] .= id_full_mat[:, idf.sk]*idf.T
#     iderr = norm(id_full_mat - truth_mat)/norm(truth_mat)
#     idrnk = length(idf.sk)
#     println("ID Error: ", iderr)


#
#     # Print various observed stats
#     N = num_points
#     R = size(U_mat,2)
#     rnk_star = max(rank(U_mat, rtol=error), rank(V_mat, rtol=error))
#     println("N=", N, ", R=",R, ", r*=", rnk_star, ", r=", idrnk)
#     @printf("Factor:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*R*rnk_star, N*N*idrnk, N*idrnk)
#     @printf("Apply:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*rnk_star, N*idrnk, N*idrnk)

# println(to)

# #
end # module rank_compare
