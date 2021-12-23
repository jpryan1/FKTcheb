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

spread_param = 6
num_points = 1000

dct_n         = 100 # Iterations for discrete cosine transform
d             = 3
kern          = 1 / (1+r^2)
mat_kern(x,y) = 1 / (1+norm(x-y)^2)
lkern         = lambdify(kern)
to            = TimerOutput()

dist_name = "normal"
# centers = [rand(d) / spread_param for _ in 1:10]

# x_vecs = copy(centers)
#
# sizes = [800 100 50 10 10 10 10 5 3 2]
# for (sz_idx, c) in enumerate(centers)
#     for i in 1:(sizes[sz_idx]-1)
#         push!(x_vecs, c + (randn(d)/(20spread_param)))
#     end
# end
#
# xs = []
# ys = []
# for v in x_vecs
#     push!(xs, v[1])
#     push!(ys, v[2])
# end
# p = scatter(xs, ys)
# display(p)


dist = Normal(0,1)
x_vecs = [rand(dist,d) / spread_param for _ in 1:num_points]

# for i in 1:num_points
#     x_vecs[i] ./= norm(x_vecs[i])
#     x_vecs[i][1] = 0.8sqrt(1-norm(x_vecs[i][2:end])^2)
#     # x_vecs[i] .*= 0.5
# end
# for i in 1:50
#     push!(x_vecs, (randn(d)/8) .+ 0.5)
# end

truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
max_norm = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
println("Max norm: ",max_norm)

svecs, svals = svd(truth_mat);

fkt_ranks = []
fkt_errs = []

# for tolpow in 3:2:7
for tolpow in 0.5:3.5
    new_fkt_ranks = []
    new_fkt_errs = []
    for fkt_deg in 2:2:20
        println("FKT test: ",tolpow, " ", fkt_deg)
        rtol = 10.0^(-tolpow)

        cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to, rtol)
        U_mat = degen_kern_harmonic(lkern, x_vecs, cfg)
        V_mat = transpose(U_mat)
        fkt_rank = size(V_mat, 1)
        fkt_guess = U_mat*V_mat
        # _, fkterrorsvals = svd(fkt_guess-truth_mat)
        fkt_err_2norm = norm(fkt_guess-truth_mat, 2)
        push!(new_fkt_ranks, fkt_rank)
        # push!(new_fkt_errs, fkterrorsvals[1]/svals[1])
        push!(new_fkt_errs, fkt_err_2norm/svals[1])
    end
    push!(fkt_ranks, new_fkt_ranks)
    push!(fkt_errs, new_fkt_errs)
end

nystrom_ranks = []
nystrom_errs = []

for idrnk in 1:10:150
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
    nystrom_err_2norm = norm(nystrom_guess-truth_mat,2)
    push!(nystrom_ranks, idrnk)
    push!(nystrom_errs, nystrom_err_2norm/svals[1])
    # println("Nystrom r=",idrnk, ", err=", nystromerror)
end
p = plot(2:151, svals[2:151]/svals[1], yaxis=:log10, label="SVD")
p = plot!(nystrom_ranks, nystrom_errs, label="Nystrom")
for i in 1:length(fkt_ranks)
    p = scatter!(fkt_ranks[i], fkt_errs[i], label=string("FKT ", i))
end

ytick_vals = [10.0^(-i) for i in 1:10]
xtick_vals = collect(0:50:150)
p = scatter!([0],[1], label=false, legend=:bottomright, yticks=(ytick_vals,ytick_vals), xticks=(xtick_vals,xtick_vals))
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

#
end # module rank_compare
