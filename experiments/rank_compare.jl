module rank_compare

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random
using Plots

r = Sym("r")

fn_approx_plot = true

dct_n         = 100 # Iterations for discrete cosine transform
d             = 3
kern          = 1 / (1+r^2)
mat_kern(x,y) = 1 / (1+norm(x-y)^2)
lkern         = lambdify(kern)
to            = TimerOutput()

num_points = 1000
println("\n\nN=",num_points)
x_vecs = [randn(d) / 2 for _ in 1:num_points]


truth_mat  = mat_kern.(x_vecs, permutedims(x_vecs))
max_norm = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
println("Max norm: ",max_norm)

rtol = 1e-16
rs = []
subset = rand(length(x_vecs),length(x_vecs))
for (x_idx1, x_vec1) in enumerate(x_vecs)
    for (x_idx2, x_vec2) in enumerate(x_vecs)
        if subset[x_idx1,x_idx2] > 0.01
            continue
        end
        if x_idx1 > x_idx2
            continue
        end
        push!(rs, norm(x_vec1-x_vec2))
    end
end

xs = collect(0:0.1:maximum(rs))
truths =lkern.(xs)
p = plot(xs, truths, label="Truth")
# fkt_deg = 6
for fkt_deg in 12
    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to, rtol)
    U_mat, V_mat = degen_kern_harmonic(lkern, x_vecs, x_vecs, cfg)
    guess = U_mat*V_mat
    ys = []
    for (x_idx1, x_vec1) in enumerate(x_vecs)
        for (x_idx2, x_vec2) in enumerate(x_vecs)
            if subset[x_idx1,x_idx2] > 0.01
                continue
            end
            if x_idx1 > x_idx2
                continue
            end
            push!(ys, real(guess[x_idx1, x_idx2]))
        end
    end
    p = scatter!(rs, ys, label=string(fkt_deg))
end
display(p)

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


end # module rank_compare
