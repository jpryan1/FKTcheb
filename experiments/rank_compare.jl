module rank_compare

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random

r = Sym("r")

dct_n         = 100 # Iterations for discrete cosine transform
d             = 3

kern          = 1 / (1+r^2)
mat_kern(x,y) = 1 / (1+norm(x-y)^2)
# kern          = exp(-abs(r))
# mat_kern(x,y) = exp(-norm(x-y))
# kern          = 1/(0.001+abs(r))
# mat_kern(x,y) = 1/(0.001+norm(x-y))
# kern          = (1+abs(r))*exp(-abs(r))
# mat_kern(x,y) = (1+norm(x-y))*exp(-norm(x-y))
lkern         = lambdify(kern)


to      = TimerOutput()

fkt_deg = 10
# num_points = 2000
for num_points in [1_500 3_000 6_000]
    println("\n\n",num_points)
    x_vecs = [randn(d)./8 for _ in 1:num_points]
    # y_vecs = [randn(d)./8 for _ in 1:num_points]
    max_norm = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
    truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))


    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

    @timeit to string("FKT ",num_points) U_mat, V_mat = cauchy_fkt(lkern, x_vecs, x_vecs,truth_mat, cfg)

    guess = U_mat*V_mat
    error = norm(guess-truth_mat)/norm(truth_mat)
    println("FKT Error: ", error)

    @timeit to string("FKT+ IDs ", num_points) begin
        U_mat_id = idfact(U_mat, rtol = error)
        V_mat_id = idfact(V_mat, rtol = error)
    end

    U_mat[:, U_mat_id.rd] .= U_mat[:, U_mat_id.sk]*U_mat_id.T
    V_mat[:, V_mat_id.rd] .= V_mat[:, V_mat_id.sk]*V_mat_id.T
    guess = U_mat*V_mat
    errorplus = norm(guess-truth_mat)/norm(truth_mat)
    println("FKT+ Error: ", errorplus)


    # Interpolative decomposition w/ same error

    @timeit to string("ID ",num_points) begin
        id_full_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
        idf = idfact(id_full_mat, rtol = error)
    end
    id_full_mat[:, idf.rd] .= id_full_mat[:, idf.sk]*idf.T

    iderr = norm(id_full_mat - truth_mat)/norm(truth_mat)
    idrnk = length(idf.sk)
    println("ID Error: ", iderr)

    q_set = randperm(num_points)[1:idrnk]
    @timeit to string("Nystrom ", num_points) begin
        Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
        qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
    end
    nystromguess = Nq * (qmat \ transpose(Nq))
    for i in 1:length(num_points)
        nystromguess[i,i] = truth_mat[i,i]
    end

    nystromerror = norm(nystromguess-truth_mat)/norm(truth_mat)
    println("Nystrom Error: ", nystromerror)


    N = num_points
    R = size(U_mat,2)
    rnk_star = max(rank(U_mat, rtol=error), rank(V_mat, rtol=error))
    println("N=", N, ", R=",R, ", r*=", rnk_star, ", r=", idrnk)

    @printf("Factor:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*R*rnk_star, N*N*idrnk, N*idrnk)
    @printf("Apply:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*rnk_star, N*idrnk, N*idrnk)
    # println(rnk_star/idrnk," <? ",N/R)
end
println(to)
#
#     r = size(umat, 2)
#     push!(ranks, r)
#     @timeit idto string("n=",num_points) idf = idfact(truth_mat, rank = r)
#     approx_mat = deepcopy(truth_mat)
# #     approx_mat[:, idf.sk] .= truth_mat[:, idf.sk]
#     approx_mat[:, idf.rd] .= truth_mat[:, idf.sk]*idf.T
#     push!(iderrs, (norm(approx_mat - truth_mat))/norm(truth_mat))
# end

# test_vec   = randn(num_points)
# test_truth = truth_mat*test_vec
# truth_vec = []
# r_vec     = []
# for (x_idx, x_vec) in enumerate(x_vecs)
#     for (y_idx, y_vec) in enumerate(y_vecs)
#         push!(truth_vec, truth_mat[x_idx, y_idx])
#         push!(r_vec, norm(x_vec-y_vec))
#     end
# end
# errors     = []
# adj_errors = []
# ranks      = []
# guesses    = []
# U_mats = []
# V_mats = []



end # module rank_compare
