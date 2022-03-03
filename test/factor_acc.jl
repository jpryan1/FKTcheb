module FactorTest

using Test

using FKTcheb
using SymPy
using TimerOutputs
using LinearAlgebra
using LowRankApprox
using Combinatorics
using SpecialFunctions
using Polynomials
using Printf
using Random
using Plots

r = Sym("r")

rtol          = 1e-3
num_points    = 1000

dct_n         = 100 # Iterations for discrete cosine transform

# lkern(r)      = r^2
# lkern(r)      = exp(1-r^2)
σ = 3
lkern(r)      = 1/(1+(σ*r)^2)
# lkern(r)      = 1/(1+r^4)
mat_kern(x,y) = lkern(norm(x-y))


# y_vecs = [randn(d)./8 for _ in 1:num_points]
@testset "3D" begin
    d         = 3
    to        = TimerOutput()
    x_vecs    = [randn(d) for _ in 1:num_points]
    mn = maximum(norm.(x_vecs))
    for i in 1:length(x_vecs)
        x_vecs[i] /= (mn)
    end
    truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
    println("True rank ", rank(truth_mat, rtol=1e-6))
    # Perform FKTcheb
    U_mat, diag = degen_kern_harmonic(lkern, x_vecs, rtol,to)
    V_mat = transpose(U_mat)
    guess = U_mat*diagm(diag)*V_mat
    println("err = ", norm(guess-truth_mat)/norm(truth_mat))
    @test isapprox(guess, truth_mat, rtol = 50rtol)
end

@testset "4D" begin
    d         = 4
    to        = TimerOutput()
    x_vecs    = [randn(d) for _ in 1:num_points]
    mn = maximum(norm.(x_vecs))
    for i in 1:length(x_vecs)
        x_vecs[i] /= (mn)
    end
     truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))
    # Perform FKTcheb
    U_mat, diag = degen_kern_harmonic(lkern, x_vecs, rtol,to)
    V_mat = transpose(U_mat)
    guess = U_mat*diagm(diag)*V_mat
    println("err = ", norm(guess-truth_mat)/norm(truth_mat))
    @test isapprox(guess, truth_mat, rtol = 50rtol)
end


@testset "2D" begin
    d         = 2
    to        = TimerOutput()
    x_vecs    = [randn(d) for _ in 1:num_points]
    mn = maximum(norm.(x_vecs))
    for i in 1:length(x_vecs)
        x_vecs[i] /= (mn)
    end
        truth_mat = mat_kern.(x_vecs, permutedims(x_vecs))

    # Perform FKTcheb
    U_mat, diag = degen_kern_harmonic(lkern, x_vecs, rtol,to)
    V_mat = transpose(U_mat)
    guess = U_mat*diagm(diag)*V_mat
    println("err = ", norm(guess-truth_mat)/norm(truth_mat))
    @test isapprox(guess, truth_mat, rtol = 50rtol)
end

#
# #
# # #

# @testset "single_eval" begin
    # d = 3
    # deg = 20
    # alpha = (d//2)-1
    # x_vec = randn(d)/4
    # y_vec = randn(d)/4
    # b = 2norm(x_vec-y_vec)
    # pij    = get_pij_table(deg+1)
    # a_vals = zeros(deg+1) # kern's coefs in cheb poly basis
    # for i in 0:(deg)
    #     a_vals[i+1] = dct(lkern, i, b, dct_n)
    # end
    # # a_vals[1]/=2
    # trans_table = get_trans_table(deg, d, b, a_vals, pij)
    #
    # tot = 0
    # normalizer_table = hyper_normalizer_table(d, convert(Int64, deg/2))
    # x_vecs = [x_vec]
    # y_vecs = [y_vec]
    # rj_hyps = cart2hyp.(x_vecs)
    # ry_hyps = cart2hyp.(y_vecs)
    # U_vals = []
    # V_vals = []
    # for k in 0:convert(Int64, deg/2)
    #
    #     mis = get_multiindices(d, k)
    #     x_harmonics = Array{Float64,2}(undef, length(x_vecs),length(mis))
    #     x_harmonics .= hyperspherical.(rj_hyps, k, permutedims(mis), Val(false))
    #     x_harmonics .= (x_harmonics ./ normalizer_table[k+1, 1:size(x_harmonics,2)]')
    #
    #     y_harmonics = Array{Float64,2}(undef, length(y_vecs),length(mis))
    #     y_harmonics .= hyperspherical.(ry_hyps, k, permutedims(mis), Val(false))
    #     y_harmonics .= (y_harmonics ./ normalizer_table[k+1, 1:size(y_harmonics,2)]')
    #
    #
    #     cosang = dot(x_vec, y_vec)/(norm(x_vec)*(norm(y_vec)))
    #     geg_poly = gegenbauer((d//2)-1, k, cosang)
    #     for harmonic_ord in 1:size(x_harmonics, 2)
    #         for m in k:2:(deg-k)
    #             push!(U_vals,(norm(y_vec)^m)*y_harmonics[1, harmonic_ord])
    #             vval = 0
    #             for n in k:2:(deg-m)
    #                 vval += ((norm(x_vec)^n)
    #                                 *trans_table[(k, m, n)]
    #                                 *x_harmonics[1, harmonic_ord])
    #             end
    #             push!(V_vals,vval)
    #         end
    #     end
    # end
    # tot = dot(U_vals, V_vals)
    # # cp = ChebyshevT(a_vals)
    # # rs = collect(0:0.01:b)
    #
    # # for i in 0:deg
    # #     for j in 0:2:i
    # #         for k3 in 0:convert(Int64, j/2)
    # #             for k2 in 0:(j-k3)
    # # for j in 0:convert(Int64, deg/2)
    # #     for m in j:2:(deg-j)
    # #         for n in j:2:(deg-m)
    # #             table_val = 0
    # #             global tot += (trans_table[(j,m,n)] * (norm(x_vec)^(n))
    # #             *(norm(y_vec)^(m)) * gegenbauer(d//2-1, j, (dot(x_vec,y_vec)/(norm(x_vec)*norm(y_vec)))))
    # #         end
    # #     end
    # #     # end
    # # end
    # println(norm(tot - lkern.(norm(x_vec-y_vec)))/norm(lkern.(norm(x_vec-y_vec))))

    # p=plot(rs, tot)
    # p=plot!(rs, lkern.(rs))
    # display(p)
    # for k3 in 0:convert(Int64, deg/2)
    #     for k2 in 0:(convert(Int64, deg/2)-k3)
    #         for k1 in 0:(convert(Int64, deg/2)-k3-k2)
    #             for i in 2(k1+k2+k3):deg
    #                 tot += (a_vals[i+1]
    #                     * (1-delta(0,i)/2)
    #                     *multinomial([k1, k2, k3]...)
    #                     *((1.0/b)^(2(k1+k2+k3)))
    #                     *(dot(x_vec,y_vec)^k3)
    #                     *norm(y_vec)^(2k2)
    #                     *norm(x_vec)^(2k1)
    #                     *pij[i+1, 2(k1+k2+k3)+1]
    #                     *(-2)^k3
    #                     )
    #                 end
    #             end
    #         end
    #     end

    # println(tot)
    # println(lkern(norm(x_vec-y_vec)))
# end
# # #
# # # @testset "gegen2" begin
# # #     d = 2
# # #     alpha = (d//2)-1
# # #     x_vec = randn(d)
# # #     y_vec = randn(d)
# # #     cosgamma = dot(x_vec,y_vec)/(norm(x_vec)*norm(y_vec))
# # #
# # #     for k in 0:10
# # #         guess1 = cosgamma^k
# # #         guess2 = 0
# # #         for j in 0:k
# # #             guess2 += chebyshev(j, cosgamma) * A(j,k, alpha)
# # #         end
# # #         @test isapprox(guess1, guess2, rtol = 1e-6)
# # #     end
# # # end
#
# #
end # module rank_compare
