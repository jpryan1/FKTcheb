module FactorTest

using Test

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Printf
using Random

r = Sym("r")

rtol          = 1e-3
num_points    = 50

dct_n         = 100 # Iterations for discrete cosine transform
kern          = 1 / (1+r^2)
lkern         = lambdify(kern)
to            = TimerOutput()
mat_kern(x,y) = 1 / (1+norm(x-y)^2)

# for idx in 1:length(x_vecs)
#     x_vecs[idx][1] = abs(x_vecs[idx][1])
#     x_vecs[idx][2:end] .= 0
# end
# y_vecs = [randn(d)./8 for _ in 1:num_points]
@testset "3D" begin
    d         = 3
    fkt_deg   = 10
    x_vecs    = [randn(d) / 8 for _ in 1:num_points]
    truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
    max_norm  = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

    # Perform FKTcheb
    U_mat, V_mat = degen_kern_harmonic(lkern, x_vecs, x_vecs, cfg)

    # Get FKTcheb error
    @test isapprox(real(U_mat*V_mat), truth_mat, rtol = rtol)
end

@testset "4D" begin
    d         = 4
    fkt_deg   = 10
    x_vecs    = [randn(d) / 8 for _ in 1:num_points]
    truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
    max_norm  = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

    # Perform FKTcheb
    U_mat, V_mat = degen_kern_harmonic(lkern, x_vecs, x_vecs, cfg)

    # Get FKTcheb error
    @test isapprox(real(U_mat*V_mat), truth_mat, rtol = rtol)
end


@testset "2D" begin
    d         = 2
    fkt_deg   = 10
    x_vecs    = [randn(d) / 8 for _ in 1:num_points]
    truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
    max_norm  = max(maximum(norm.(x_vecs)), maximum(norm.(x_vecs)))
    cfg = fkt_config(fkt_deg, d, 2max_norm, dct_n, to)

    # Perform FKTcheb
    U_mat, V_mat = degen_kern_harmonic(lkern, x_vecs, x_vecs, cfg)

    # Get FKTcheb error
    @test isapprox(real(U_mat*V_mat), truth_mat, rtol = rtol)
end



#
# @testset "gegen3" begin
#     d = 3
#     alpha = (d//2)-1
#     x_vec = randn(d)
#     y_vec = randn(d)
#     cosgamma = dot(x_vec,y_vec)/(norm(x_vec)*norm(y_vec))
#
#     for k in 0:5
#         guess1 = cosgamma^k
#         guess2 = 0
#         for j in 0:k
#             guess2 += gegenbauer(alpha, j, cosgamma) * A(j,k, alpha)
#         end
#         @test isapprox(guess1, guess2, rtol = 1e-6)
#     end
# end
#
# @testset "gegen2" begin
#     d = 2
#     alpha = (d//2)-1
#     x_vec = randn(d)
#     y_vec = randn(d)
#     cosgamma = dot(x_vec,y_vec)/(norm(x_vec)*norm(y_vec))
#
#     for k in 0:10
#         guess1 = cosgamma^k
#         guess2 = 0
#         for j in 0:k
#             guess2 += chebyshev(j, cosgamma) * A(j,k, alpha)
#         end
#         @test isapprox(guess1, guess2, rtol = 1e-6)
#     end
# end


end # module rank_compare
