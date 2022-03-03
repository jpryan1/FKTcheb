module FactorTest
using LinearAlgebra
using SpecialPolynomials


function falling(x,r)
    if r==0 return 1 end
    prod = 1
    println(r)
    for i in 0:(r-1)
        prod *= (x+i)
    end
    return prod
end


function Ar(r)
    if r==0 return 1 end
    prod = 1
    for i in r:-1:1
        prod *= (0.5+r-i)/i
    end
    return prod
end


function heart(p,q,r)
    return ((Ar(r)*Ar(p-r)*Ar(q-r)*(2p+2q-4r+1))/(Ar(p+q-r)*(2p+2q-2r+1)))
end

function multiply_polys(a::AbstractVector, b::AbstractVector)

    ans = zeros(length(a))
    adeg = 0
    for i in length(a):-1:1
        if a[i] != 0
            adeg=i
            break
        end
    end
    bdeg = 0
    for i in length(b):-1:1
        if b[i] != 0
            bdeg=i
            break
        end
    end
    if adeg+bdeg > length(a)
        println("ERROR DROPPING TERMS") # DO NOT DELETE UNLESS TEMPORARILY FOR TIMING
    end
    # for i in 1:length(a)
    #     for j in 1:i
    #         # i-1 = j-1 + i-j
    #         ans[i] += a[j]*b[i-j+1]
    #     end
    # end

    for i in 1:adeg
        for j in 1:bdeg
            if a[i]*b[j] != 0
                top = max(i,j)
                bot = min(i,j)
                p=top-1
                q=bot-1
                for r in 0:q
    # on the product of two legendre polynomials
                    ans[p+q-2r+1] += a[i]*b[j]*heart(p,q,r)
                end
            end
        end
    end

    return ans
end


function poly_mat_to_mat(pm::AbstractVector{<:AbstractVector}, data_pows::AbstractVector{<:AbstractVector}, pm_lower_deg::Int64, pm_upper_deg::Int64)
    mat = zeros(length(data_pows[1]), length(pm))
    for j in 1:length(pm)
        for i in pm_lower_deg:pm_upper_deg
            mat[:,j] .+= pm[j][i]*data_pows[i]
        end
    end
    return mat
end


function poly_col_dot(poly1::AbstractVector,
                        poly2::AbstractVector,
                            data_pow_sums::AbstractVector)
    poly_dot = multiply_polys(poly1,poly2)
    tot = 0
    for i in 1:length(poly_dot)
        # println("pair ",poly_dot[i], " ", data_pow_sums[i] )
        tot += poly_dot[i]*data_pow_sums[i]
    end
    return tot
end


function pm_mul_mat(pm::Array{Array{Float64, 1}, 1}, mat::Array{Float64, 2})
    new_polys = Array{Array{Float64,1},1}(undef, size(mat, 2))
    for j in 1:size(mat, 2)
        cur_poly = zeros(length(pm[1]))
        for i in 1:size(mat,1)
            cur_poly .+= (mat[i,j] * pm[i])
        end
        new_polys[j] = cur_poly
    end
    return new_polys
end

function pm_mul_pm(pm1::Array{Array{Float64, 1}, 1}, pm2::Array{Array{Float64, 1}, 1}, data_pow_sums::AbstractVector)
    out = zeros(length(pm1), length(pm2))
    for i in 1:size(out,1)
        for j in 1:size(out,2)
            out[i,j] = poly_col_dot(pm1[i], pm2[j], data_pow_sums)
        end
    end
    return out
end


degree = 20
polys = Array{Array{Float64, 1}, 1}()

# first=zeros(2degree+2)
# first[1]=1
# second=zeros(2degree+2)
# second[2]=1
# push!(polys, first)
# push!(polys, second)
# for i in 3:degree
#     pij = zeros(2degree+2)
#     for j in 1:i
#         if j == 1
#             pij[j] = -polys[i-2][j]
#         else
#             pij[j] = 2polys[i-1][j-1] - polys[i-2][j]
#         end
#     end
#     push!(polys, pij)
# end


for i in 1:degree
    p = zeros(2degree+2)
    p[i] = 1
    push!(polys, p)
end

# x_vecs = [randn(3) for i in 1:10000]
# mn = maximum(norm.(x_vecs))
# for i in 1:length(x_vecs)
#     x_vecs[i] /= mn
# end

x_vec_norms = [rand() for _ in 1:10000]
data_pows = Array{Array{Float64,1}, 1}()
for i in 1:(2degree+2)
    leg = zeros(i)
    leg[i] = 1
    legpol = Legendre(leg)
    push!(data_pows, legpol.(2x_vec_norms .- 1))
end
Atrue = poly_mat_to_mat(polys, data_pows, 1, length(data_pows))

function poly_qr(polys, x_vec_norms, data_pows)
    k_to_data_pow_sums = Dict()
    for k in 1:length(polys)
        x_vec_norms_k = x_vec_norms[k:end]
        data_pows_k = Array{Array{Float64,1}, 1}()
        for i in 1:(2degree+2)
            leg = zeros(i)
            leg[i] = 1
            legpol = Legendre(leg)
            push!(data_pows_k, legpol.(2x_vec_norms_k .- 1))
        end
        data_pow_sums_k = [sum(data_pows_k[i]) for i in 1:length(data_pows_k)]
        k_to_data_pow_sums[k] = data_pow_sums_k
    end
    # R = zeros(length(polys), length(polys))
    R = zeros(length(x_vec_norms), length(polys))
    vs = []
    for k in 1:(length(polys))
        x_poly = polys[k]
        xn = sqrt(poly_col_dot(x_poly,x_poly,k_to_data_pow_sums[k]))

        x1 = Legendre(x_poly)(2x_vec_norms[k]-1)
        sn = sign(x1)
        alpha = sn*xn
        vkn = sqrt(xn^2 + alpha^2 + 2alpha*x1)

        vk = poly_mat_to_mat([x_poly],  data_pows, 1, length(data_pows))[k:end, 1]
        vk[1] += alpha
        push!(vs, vk/vkn)

        tvec = zeros(length(polys)-k)
        for i in (k+1):length(polys)
            tvec[i-k] = poly_col_dot(x_poly, polys[i], k_to_data_pow_sums[k])
        end
        row_vec = [Legendre(p)(2x_vec_norms[k]-1) for p in polys[(k+1):end]]
        right_mat = transpose(tvec)/vkn^2 + ((alpha*transpose(row_vec))/(vkn^2))

        R[k,k] = -sn*xn
        R[k, (k+1):end] .= row_vec .- transpose(2*(x1+alpha)*right_mat)

        for j in (k+1):length(polys)
            polys[j] .-= 2*right_mat[1, j-k]*polys[k]
        end
    end
    return vs, R
end

function q_poly_mul(vs, B)
    C = copy(B)
    for k in (length(vs)):-1:1
        C[k:end, :] -= 2*(vs[k]*(transpose(vs[k])*C[k:end,:]))
    end
    return C
end

function qtrans_poly_mul(vs, B)
    C = copy(B)
    for k in 1:length(vs)
        C[k:end, :] -= 2*(vs[k]*(transpose(vs[k])*C[k:end,:]))
    end
    return C
end


vs, R = poly_qr(polys, x_vec_norms, data_pows)
A = q_poly_mul(vs, R)
println(size(Atrue), " ", size(A))
println(norm(A-Atrue,2)/norm(A,2))
# println(cond(B))

#
end
