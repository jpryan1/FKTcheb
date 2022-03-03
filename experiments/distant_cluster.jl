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

num_points = 1000

dct_n         = 100 # Iterations for discrete cosine transform
d             = 3
σ = 5
lkern(r)      = 1/(1+(σ*r)^2)
# lkern(r)      = 1/(1+r^4)
mat_kern(x,y) = lkern(norm(x-y))
to            = TimerOutput()

x_vecs    = [randn(d) for _ in 1:num_points]
mn = maximum(norm.(x_vecs))
for i in 1:length(x_vecs)
    x_vecs[i] /= (mn)
end


# dist = Normal(0,1)
# x_vecs = [rand(dist,d) / spread_param for _ in 1:num_points]
#TODO Pareto fails for uniform dist

# for i in 1:num_points
#     x_vecs[i] ./= norm(x_vecs[i])
#     x_vecs[i][1] = 0.9sqrt(1-norm(x_vecs[i][2:end])^2)
#     x_vecs[i] .*= 0.5
# end
# for i in 1:50
#     push!(x_vecs, (randn(d)/8) .+ 0.5)
# end

truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))

#
svecs, svals = svd(truth_mat);

fkt_ranks = []
fkt_errs = []

# for tolpow in 1:2:5
#
#     new_fkt_ranks = []
#     new_fkt_errs = []
#     for fkt_deg in 2:2:10
#         println("FKT test: ",tolpow, " ", fkt_deg)
#         rtol = 10.0^(-tolpow)
#
#         U_mat = degen_kern_harmonic(lkern, x_vecs, rtol,to)
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
for rtol_pow in 0:6
    rtol = (1e-3)*(2.0^rtol_pow)

    U_mat = degen_kern_harmonic(lkern, x_vecs, rtol,to)
    V_mat = transpose(U_mat)
    fkt_rank = size(V_mat, 1)
    # if fkt_rank > 100
    #     break
    # end
    fkt_guess = U_mat*V_mat
    # _, fkterrorsvals = svd(fkt_guess-truth_mat)
    fkt_err_2norm = norm(fkt_guess-truth_mat, 2)
    push!(new_fkt_ranks, fkt_rank)
    # push!(new_fkt_errs, fkterrorsvals[1]/svals[1])
    push!(new_fkt_errs, fkt_err_2norm/svals[1])
    # push!(new_fkt_errs, rtol)
    # println("Pareto ", fkt_deg, " err: ", fkt_err_2norm/svals[1], " rank ", fkt_rank, " RTOL: ", rtol)
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

pows = [1, 3, 5, 7, "Adaptive"]
p = plot(1:100, svals[2:101]/svals[1], yaxis=:log10, label="SVD")
p = plot!(nystrom_ranks, nystrom_errs, label="Nystrom")
# for i in 1:(length(fkt_ranks)-1)
#     p = scatter!(fkt_ranks[i], fkt_errs[i], label=string("FKT ", pows[i]))
# end
println(fkt_ranks[end])
println(fkt_errs[end])
p = scatter!(fkt_ranks[end], fkt_errs[end], label=string("Pareto"),markershape=:hexagon)

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
