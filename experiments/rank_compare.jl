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
    println("\n\nN=",num_points)
    x_vecs = [randn(d)s for _ in 1:num_points]
    # y_vecs = [randn(d)./8 for _ in 1:num_points]
    truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))

    max_norm = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

    # Perform FKTcheb
    @timeit to string("FKT ",num_points) U_mat, V_mat = cauchy_fkt(lkern, x_vecs, x_vecs,truth_mat, cfg)

    # Get FKTcheb error
    guess = U_mat*V_mat
    error = norm(guess-truth_mat)/norm(truth_mat)
    println("FKT Error: ", error)

    # Perform IDs for FKTcheb+
    @timeit to string("FKT+ IDs ", num_points) begin
        U_mat_id = idfact(U_mat, rtol = error)
        V_mat_id = idfact(V_mat, rtol = error)
    end

    # Get errpr for FKTcheb+
    U_mat[:, U_mat_id.rd] .= U_mat[:, U_mat_id.sk]*U_mat_id.T
    V_mat[:, V_mat_id.rd] .= V_mat[:, V_mat_id.sk]*V_mat_id.T
    guess = U_mat*V_mat
    errorplus = norm(guess-truth_mat)/norm(truth_mat)
    println("FKT+ Error: ", errorplus)


    # Perform ID with tolerance for same error as FKTcheb
    @timeit to string("ID ",num_points) begin
        id_full_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
        idf = idfact(id_full_mat, rtol = error)
    end

    # Get error for ID
    id_full_mat[:, idf.rd] .= id_full_mat[:, idf.sk]*idf.T
    iderr = norm(id_full_mat - truth_mat)/norm(truth_mat)
    idrnk = length(idf.sk)
    println("ID Error: ", iderr)

    # Perform Nystrom with num points equal to rank seen in ID
    q_set = randperm(num_points)[1:idrnk]
    @timeit to string("Nystrom ", num_points) begin
        Nq =  mat_kern.(x_vecs, permutedims(x_vecs[q_set]))
        qmat = lu( mat_kern.(x_vecs[q_set], permutedims(x_vecs[q_set])))
    end

    # Get error for Nystrom
    nystromguess = Nq * (qmat \ transpose(Nq))
    for i in 1:length(num_points)
        nystromguess[i,i] = truth_mat[i,i]
    end
    nystromerror = norm(nystromguess-truth_mat)/norm(truth_mat)
    println("Nystrom Error: ", nystromerror)

    # Print various observed stats
    N = num_points
    R = size(U_mat,2)
    rnk_star = max(rank(U_mat, rtol=error), rank(V_mat, rtol=error))
    println("N=", N, ", R=",R, ", r*=", rnk_star, ", r=", idrnk)
    @printf("Factor:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*R*rnk_star, N*N*idrnk, N*idrnk)
    @printf("Apply:\n FKT:%10d\nFKT+:%10d\n  ID:%10d\n Nys:%10d\n", N*R, N*rnk_star, N*idrnk, N*idrnk)
end

println(to)


end # module rank_compare
